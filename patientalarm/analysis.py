import argparse
import json
from pathlib import Path
import numpy as np
from .common import bootstrap, ratio


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/metrics.json")
    parser.add_argument("--output", default="outputs/patient_analysis.json")
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    result = {}
    for policy, record in data["policies"].items():
        rows = record.get("patients", [])
        if not rows:
            continue
        ids = np.array([r["patient"] for r in rows])

        def sensitivity(ix):
            return ratio(
                sum((rows[i]["timely"] for i in ix)),
                sum((rows[i]["events"] for i in ix)),
            )

        def burden(ix):
            return float(
                np.mean(
                    [
                        rows[i]["rate_per_day"]
                        for i in ix
                        if rows[i]["rate_per_day"] is not None
                    ]
                )
            )

        result[policy] = {
            "s30_ci95": bootstrap(ids, sensitivity, args.draws, 11),
            "mean_f_ci95": bootstrap(ids, burden, args.draws, 11),
        }
    Path(args.output).write_text(
        json.dumps(json_ready(result), indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
