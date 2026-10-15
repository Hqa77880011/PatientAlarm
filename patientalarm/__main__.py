import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, firwin, lfilter, sosfilt
from scipy.stats import beta
from torch import nn
from torch.nn import functional as F
from .common import binary_metrics, check_groups, fit, probabilities, ratio, temperature


def json_ready(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    return value


MONTAGE = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FZ-CZ",
    "CZ-PZ",
]


class EEGNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding="same", bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (18, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), groups=16, padding="same", bias=False),
            nn.Conv2d(16, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        return self.head(self.features(x[:, None]).flatten(1)).squeeze(1)


def prepare(args, cfg):
    import mne

    manifest = pd.read_csv(args.manifest, dtype={"patient": str})
    manifest["patient"] = manifest.patient.replace({"chb21": "chb01"})
    check_groups(manifest.patient.to_numpy(), manifest.split.to_numpy())
    events = pd.read_csv(args.events, dtype={"patient": str})
    events["patient"] = events.patient.replace({"chb21": "chb01"})
    examples = []
    segments = []
    acquired = []
    for patient, group in manifest.groupby("patient"):
        joined = []
        segment_start = None
        last_end = None
        last_fs = None

        def finish():
            if not joined:
                return
            values = np.concatenate(joined, axis=1)
            fs = last_fs
            if fs == 512:
                values = lfilter(
                    firwin(63, 0.45, window=("kaiser", 5)), [1.0], values, axis=1
                )[:, ::2]
            elif fs != 256:
                raise ValueError(
                    "Supported causal acquisition rates are 256 and 512 Hz"
                )
            values = sosfilt(
                butter(4, [0.5, 40], btype="bandpass", fs=256, output="sos"),
                values,
                axis=1,
            )
            end = segment_start + values.shape[1] / 256
            segments.append(
                {
                    "patient": patient,
                    "start": segment_start + 8,
                    "end": end,
                    "split": group.iloc[0].split,
                }
            )
            reference = events[events.patient == patient]
            for endpoint in range(8 * 256, values.shape[1] + 1, 256):
                t = segment_start + endpoint / 256
                y = int(((reference.start <= t) & (t <= reference.end)).any())
                examples.append(
                    (
                        values[:, endpoint - 1024 : endpoint].astype(np.float32),
                        y,
                        patient,
                        t,
                        group.iloc[0].split,
                    )
                )

        for row in group.sort_values("start").itertuples(index=False):
            raw = mne.io.read_raw_edf(row.file, preload=True, verbose="ERROR")
            fs = int(raw.info["sfreq"])
            acquired.append(
                {
                    "patient": patient,
                    "start": row.start,
                    "end": row.start + raw.n_times / fs,
                    "split": row.split,
                }
            )
            names = {
                c.upper().replace("EEG ", "").strip(): i
                for i, c in enumerate(raw.ch_names)
            }
            if not all((channel in names for channel in MONTAGE)):
                finish()
                joined = []
                last_end = None
                segment_start = None
                continue
            values = raw.get_data(picks=[names[c] for c in MONTAGE]) * 1000000.0
            if not np.isfinite(values).all():
                raise ValueError(
                    "Split unreadable/nonfinite intervals into unavailable manifest segments"
                )
            if last_end is None or abs(row.start - last_end) > 1 / fs or fs != last_fs:
                finish()
                joined = []
                segment_start = float(row.start)
            joined.append(values)
            last_fs = fs
            last_end = row.start + values.shape[1] / fs
        finish()
    target = Path(args.data)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        **{
            name: np.asarray([r[i] for r in examples])
            for i, name in enumerate(["x", "y", "patient", "time", "split"])
        }
    )
    target.with_suffix(".json").write_text(
        json.dumps(
            json_ready(
                {
                    "segments": segments,
                    "acquired": acquired,
                    "events": events.to_dict("records"),
                    "fir_delay_s_512hz": 31 / 512,
                    "montage": MONTAGE,
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )


def emit(times, scores, threshold, refractory=30):
    notifications = []
    active = False
    last_positive = -np.inf
    last_emission = -np.inf
    persistence = 0
    previous = None
    for t, p in zip(times, scores):
        if previous is not None and t - previous > 1.01:
            active = False
            persistence = 0
        previous = t
        if active:
            if p >= threshold:
                last_positive = t
            if t - last_positive <= 5:
                continue
            active = False
        if t - last_emission < refractory:
            persistence = 0
            continue
        persistence = persistence + 1 if p >= threshold else 0
        if persistence >= 3:
            notifications.append(float(t))
            last_emission = last_positive = t
            active = True
            persistence = 0
    return notifications


def summarize(d, meta, indices, probs, threshold, reference_all=False, refractory=30):
    patient_rows = []
    all_delays = []
    alarms = {}
    roles = set(d["split"][indices])
    identities = sorted(
        set(d["patient"][indices])
        | {r["patient"] for r in meta["acquired"] if r["split"] in roles}
    )
    for patient in identities:
        ix = indices[d["patient"][indices] == patient]
        order = np.argsort(d["time"][ix])
        ix = ix[order]
        positions = {int(v): i for i, v in enumerate(indices)}
        scores = np.array([probs[positions[int(j)]] for j in ix])
        times = d["time"][ix]
        notifications = emit(times, scores, threshold, refractory)
        alarms[patient] = notifications
        segments = [s for s in meta["segments"] if s["patient"] == patient]
        seconds = sum((max(0, s["end"] - s["start"]) for s in segments))
        acquired = sum(
            (r["end"] - r["start"] for r in meta["acquired"] if r["patient"] == patient)
        )
        refs = sorted(
            [r for r in meta["events"] if r["patient"] == patient],
            key=lambda r: r["start"],
        )
        scorable = [
            r
            for r in refs
            if any((s["start"] <= r["start"] <= s["end"] for s in segments))
        ]
        selected = refs if reference_all else scorable
        used = set()
        delays = []
        for event in selected:
            choices = [
                j
                for j, a in enumerate(notifications)
                if j not in used and event["start"] <= a <= event["end"]
            ]
            if choices:
                j = choices[0]
                used.add(j)
                delays.append(notifications[j] - event["start"])
            else:
                delays.append(float("inf"))
        duplicate = sum(
            (
                j not in used and any((r["start"] <= a <= r["end"] for r in refs))
                for j, a in enumerate(notifications)
            )
        )
        background = len(notifications) - len(used) - duplicate
        all_delays.extend(delays)
        patient_rows.append(
            {
                "patient": patient,
                "seconds": seconds,
                "acquired_seconds": acquired,
                "events": len(selected),
                "detected": sum(np.isfinite(delays)),
                "timely": sum((x <= 30 for x in delays)),
                "background": background,
                "duplicates": duplicate,
                "nonactionable": background + duplicate,
                "rate_per_day": (
                    (background + duplicate) * 86400 / seconds if seconds else None
                ),
                "segments": len(segments),
            }
        )
    rates = [r["rate_per_day"] for r in patient_rows if r["rate_per_day"] is not None]
    events = sum((r["events"] for r in patient_rows))
    total_seconds = sum((r["seconds"] for r in patient_rows))
    metrics = {
        "sensitivity": ratio(sum((r["detected"] for r in patient_rows)), events),
        "s30": ratio(sum((r["timely"] for r in patient_rows)), events),
        "mean_f": float(np.mean(rates)) if rates else None,
        "pooled_f": ratio(
            sum((r["nonactionable"] for r in patient_rows)) * 86400, total_seconds
        ),
        "v24": float(np.mean(np.array(rates) > 24)) if rates else None,
        "median_delay_s": (
            float(np.median([v for v in all_delays if np.isfinite(v)]))
            if any(np.isfinite(all_delays))
            else None
        ),
        "availability": ratio(
            total_seconds, sum((r["acquired_seconds"] for r in patient_rows))
        ),
        "events": events,
        "patients": len(patient_rows),
        "background": sum((r["background"] for r in patient_rows)),
        "duplicates": sum((r["duplicates"] for r in patient_rows)),
    }
    return (metrics, patient_rows, alarms)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "train", "evaluate"])
    p.add_argument("--config", default="config.json")
    p.add_argument("--data", default="data/eeg.npz")
    p.add_argument("--output", default="outputs")
    p.add_argument("--manifest")
    p.add_argument("--events")
    p.add_argument("--refractory", type=float, default=30)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    torch.manual_seed(cfg["seed"])
    if args.command == "prepare":
        prepare(args, cfg)
        return
    d = np.load(args.data)
    check_groups(d["patient"], d["split"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "train":
        model = EEGNet()
        train_ix = np.flatnonzero(d["split"] == "train")
        mean = d["x"][train_ix].mean(axis=(0, 2))
        scale = d["x"][train_ix].std(axis=(0, 2)).clip(1e-06)
        np.savez(out / "normalization.npz", mean=mean, scale=scale)
        positive = float(np.sum(d["y"][train_ix]))
        pos_weight = (len(train_ix) - positive) / max(positive, 1)

        def items(split):
            ix = np.flatnonzero(d["split"] == split)
            return (
                torch.tensor((d["x"][ix] - mean[None, :, None]) / scale[None, :, None]),
                torch.tensor(d["y"][ix], dtype=torch.float32),
            )

        def objective(m, x, y):
            return (
                F.binary_cross_entropy_with_logits(
                    m(x), y, pos_weight=torch.tensor(pos_weight), reduction="sum"
                ),
                torch.tensor(len(y), dtype=torch.float32),
            )

        validation_items = items("validation")
        validation_ids = d["patient"][d["split"] == "validation"]

        def monitor(m, arrays):
            predictions = []
            with torch.no_grad():
                for batch in torch.arange(len(arrays[0])).split(cfg["batch_size"]):
                    predictions.extend(torch.sigmoid(m(arrays[0][batch])).numpy())
            predictions = np.asarray(predictions)
            targets = arrays[1].numpy()
            values = []
            for patient in np.unique(validation_ids):
                same = validation_ids == patient
                score = binary_metrics(targets[same], predictions[same])["auprc"]
                if score is not None:
                    values.append(score)
            if not values:
                raise ValueError(
                    "Patient-AUPRC checkpoint selection needs a development patient with seizures"
                )
            return -float(np.mean(values))

        fit(model, items("train"), validation_items, objective, cfg, out, monitor)
        return
    checkpoint = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    model = EEGNet()
    model.load_state_dict(checkpoint["state"])
    model.eval()
    norm = np.load(out / "normalization.npz")
    logits = []
    with torch.no_grad():
        for ix in np.array_split(
            np.arange(len(d["y"])), max(1, int(np.ceil(len(d["y"]) / 64)))
        ):
            x = (d["x"][ix] - norm["mean"][None, :, None]) / norm["scale"][
                None, :, None
            ]
            logits.extend(model(torch.tensor(x)).numpy())
    logits = np.array(logits)
    dev = np.flatnonzero(d["split"] == "validation")
    cal = np.flatnonzero(d["split"] == "calibration")
    test = np.flatnonzero(d["split"] == "test")
    if not len(cal) or not len(test):
        raise ValueError("Alarm-calibration and held-out patients are required")
    from .common import patient_weights

    temp = temperature(logits[dev], d["y"][dev], patient_weights(d["patient"][dev]))
    probs = probabilities(logits, temp)
    meta = json.loads(Path(args.data).with_suffix(".json").read_text())
    curves = []
    for raw_threshold in np.arange(0.01, 1, 0.01):
        threshold = float(
            probabilities(np.log(raw_threshold / (1 - raw_threshold)), temp)
        )
        m, patients, _ = summarize(
            d, meta, cal, probs[cal], threshold, refractory=args.refractory
        )
        predicted = probs[cal] >= threshold
        ycal = d["y"][cal]
        m["window_f1"] = ratio(
            2 * np.sum(predicted & (ycal == 1)), np.sum(predicted) + np.sum(ycal == 1)
        )
        m["threshold"] = float(threshold)
        m["raw_threshold"] = float(raw_threshold)
        m["patients_detail"] = patients
        eligible = [r for r in patients if r["seconds"] >= 3600 and r["segments"] <= 48]
        if eligible:
            rates = np.array([r["rate_per_day"] for r in eligible])
            n = len(rates)
            k = int(np.sum(rates > 24))
            alpha = 0.05 / (2 * 99)
            m["certificate"] = {
                "patients": n,
                "mean_upper": min(
                    4032,
                    float(
                        rates.mean() + 4032 * np.sqrt(np.log(2 * 99 / 0.05) / (2 * n))
                    ),
                ),
                "tail_upper": (
                    float(beta.ppf(1 - alpha, k + 1, n - k)) if k < n else 1.0
                ),
                "s30": ratio(
                    sum((r["timely"] for r in eligible)),
                    sum((r["events"] for r in eligible)),
                ),
            }
        curves.append(m)
    policies = {}
    for name in ["full", "mean-only", "pooled", "overall"]:
        feasible = [
            r
            for r in curves
            if (r["s30"] or 0) >= cfg["sensitivity_floor"]
            and r["mean_f"] is not None
            and (
                (r["pooled_f"] if name == "pooled" else r["mean_f"])
                <= cfg["mean_budget"]
            )
            and (name != "full" or r["v24"] <= cfg["tail_fraction"])
        ]
        policies[name] = (
            max(
                feasible,
                key=lambda r: (
                    (r["sensitivity"] if name == "overall" else r["s30"]) or 0,
                    r["sensitivity"] or 0,
                    -r["mean_f"],
                    r["threshold"],
                ),
            )["threshold"]
            if feasible
            else None
        )
    policies["window-f1"] = max(
        curves, key=lambda r: (r["window_f1"] or 0, r["threshold"])
    )["threshold"]
    certified = [
        r
        for r in curves
        if r.get("certificate")
        and r["certificate"]["mean_upper"] <= cfg["mean_budget"]
        and (r["certificate"]["tail_upper"] <= cfg["tail_fraction"])
        and ((r["certificate"]["s30"] or 0) >= cfg["sensitivity_floor"])
    ]
    policies["certified"] = (
        max(certified, key=lambda r: (r["certificate"]["s30"] or 0, r["threshold"]))[
            "threshold"
        ]
        if certified
        else None
    )
    result = {
        "temperature": temp,
        "policies": {},
        "empirical_curves": curves,
        "certified_feasible_candidates": len(certified),
    }
    for name, threshold in policies.items():
        if threshold is None:
            result["policies"][name] = {"feasible": False}
            continue
        m, patients, alarms = summarize(
            d, meta, test, probs[test], threshold, refractory=args.refractory
        )
        unconditional = summarize(
            d, meta, test, probs[test], threshold, True, args.refractory
        )[0]
        result["policies"][name] = {
            "threshold": threshold,
            "metrics": m,
            "all_events": unconditional,
            "patients": patients,
            "alarms": alarms,
        }
    full = next((r for r in curves if r["threshold"] == policies["full"]), None)
    if full:
        eligible = [
            r
            for r in full["patients_detail"]
            if r["seconds"] >= 3600 and r["segments"] <= 48
        ]
        n = len(eligible)
        M = 99
        L = 4032
        alpha = 0.05 / (2 * M)
        if n:
            rates = np.array([r["rate_per_day"] for r in eligible])
            k = int(np.sum(rates > 24))
            result["certificate_at_empirical"] = {
                "patients": n,
                "mean_upper": min(
                    L, float(rates.mean() + L * np.sqrt(np.log(2 * M / 0.05) / (2 * n)))
                ),
                "tail_upper": (
                    float(beta.ppf(1 - alpha, k + 1, n - k)) if k < n else 1.0
                ),
            }
    per_patient = [
        binary_metrics(
            d["y"][test][d["patient"][test] == v], probs[test][d["patient"][test] == v]
        )
        for v in np.unique(d["patient"][test])
    ]
    result["window_metrics"] = {
        key: (
            float(np.mean([r[key] for r in per_patient if r.get(key) is not None]))
            if any((r.get(key) is not None for r in per_patient))
            else None
        )
        for key in ["auroc", "auprc", "brier", "ece"]
    }
    (out / "metrics.json").write_text(
        json.dumps(json_ready(result), indent=2, allow_nan=False)
    )
    np.savez(
        out / "scores.npz",
        patient=d["patient"],
        time=d["time"],
        y=d["y"],
        p=probs,
        split=d["split"],
    )
    plt.plot([r["mean_f"] for r in curves], [r["s30"] for r in curves], ".")
    plt.axvline(cfg["mean_budget"], color="gray")
    plt.xlabel("Mean non-actionable alarms/day")
    plt.ylabel("Timely sensitivity")
    plt.tight_layout()
    plt.savefig(out / "operating_curve.png")
    plt.close()


if __name__ == "__main__":
    main()
