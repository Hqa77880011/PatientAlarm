# PatientAlarm

Implementation of patient-level alarm budgets for timely seizure detection under patient shift.

```bash
python -m pip install -e .
python -m patientalarm prepare --manifest recordings.csv --events seizures.csv
python -m patientalarm train
python -m patientalarm evaluate
```

Use [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) and an authorized [TUSZ](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml) copy for frozen transfer. `recordings.csv` has `patient,file,start,split`, where file is an EDF path, start is absolute recording time in seconds on a shared patient timeline, and split is `train`, `validation`, `calibration`, or `test`. `seizures.csv` contains global `patient,start,end` intervals, merged according to corpus annotation rules. Preserve actual gaps and same-person identity: preparation groups chb21 with chb01. Continuous same-rate EDF files are joined before preprocessing; genuine gaps reset the causal signal chain.

The primary loader requires the 18 documented bipolar derivations, explicitly named in `MONTAGE`. Unknown references or missing derivations make that recording unavailable. Signals convert to microvolts. A stateful 63-tap antialias FIR handles 512-to-256-Hz conversion; 256-Hz input bypasses it. The subsequent fourth-order Butterworth bandpass is causal. Eight-second startup and four-second trailing windows are applied once per uninterrupted segment. The sample timestamp is its physical acquisition time, so filter delay remains in event-to-emission timing. Nonfinite spans must be represented as separate unavailable recordings.

Training uses the EEGNet configuration in the paper and training-only channel moments. `config.json` controls optimizer, weighted BCE, seed, early stopping, alarm budget and sensitivity floor. Checkpoint selection maximizes mean development-patient AUPRC; weighted validation loss remains recorded separately. Evaluation calibrates probabilities on validation patients, then exhaustively replays all 99 transformed candidate thresholds on separate alarm-calibration patients. Positive temperature scaling transforms the original grid rather than creating a new policy search.

The state machine emits on the third consecutive positive, bridges up to five seconds, enforces a 30-second physical-time refractory interval and logs one emission per active plateau. Matching is chronological, one-to-one and inside the seizure interval; a post-offset notification cannot detect a short seizure. Background and duplicate notifications stay separate. Mean-patient and pooled burden use valid scorable seizure plus background time. Test outputs include event and timely sensitivity, median detected delay, patient-tail violations, available/all-event endpoints, mean-patient window AUROC/AUPRC/Brier/ECE and confidence bounds at the empirical operating point. The optional certificate remains separate from empirical policy feasibility.

`outputs/metrics.json`, `scores.npz` and `operating_curve.png` contain actual executed results. Compare the full, window-F1, mean-only, pooled and overall-objective policies on the identical scorer. `--refractory 300` provides the long-refractory ablation. Missing-channel recordings retain acquired exposure and reference events in the unconditional endpoint. Changing budgets, source, exposure or corruption requires a separate output directory and unchanged source policy for external evaluation. No stored paper table is used as a result source.

## Patient/cell analysis

Run `python -m patientalarm.analysis --draws 2000` after evaluation to compute grouped bootstrap intervals and the supported subgroup or repeatability summaries. The input/output paths can be set with `--input` and `--output`.
