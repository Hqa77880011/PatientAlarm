# PatientAlarm

Implementation of patient-level alarm budgets for timely seizure detection under patient shift.

## Installation

Use Python 3.10+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Use [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) and an authorized [TUSZ](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml) copy for frozen transfer. `recordings.csv` has `patient,file,start,split`, where file is an EDF path, start is absolute recording time in seconds on a shared patient timeline, and split is `train`, `validation`, `calibration`, or `test`. `seizures.csv` contains global `patient,start,end` intervals, merged according to corpus annotation rules. Preserve actual gaps and same-person identity: preparation groups chb21 with chb01. Continuous same-rate EDF files are joined before preprocessing; genuine gaps reset the causal signal chain.

The primary loader requires the 18 documented bipolar derivations, explicitly named in `MONTAGE`. Unknown references or missing derivations make that recording unavailable. Signals convert to microvolts. A stateful 63-tap antialias FIR handles 512-to-256-Hz conversion; 256-Hz input bypasses it. The subsequent fourth-order Butterworth bandpass is causal. Eight-second startup and four-second trailing windows are applied once per uninterrupted segment. The sample timestamp is its physical acquisition time, so filter delay remains in event-to-emission timing. Nonfinite spans must be represented as separate unavailable recordings.

```bash
python -m patientalarm prepare --manifest recordings.csv --events seizures.csv --config config.json
```

## Training and inference

```bash
python -m patientalarm train --config config.json --output outputs
```

## Evaluation

```bash
python -m patientalarm evaluate --config config.json --output outputs
python -m patientalarm.analysis --input outputs/metrics.json --output outputs/patient_analysis.json --draws 2000
```

## Outputs

`data/eeg.npz` and its JSON hold causal windows/preparation settings. `outputs/` contains model/normalization/training records, `scores.npz`, `metrics.json`, `operating_curve.png` and optional patient analysis.

Keep `config.json` with the run; it contains model/data settings and the seed where the CLI does not expose one. Use separate output directories for different seeds/conditions.

Source inspection: Python 3.13.5, numpy 2.1.3, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1, torch 2.7.1+cu118, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
