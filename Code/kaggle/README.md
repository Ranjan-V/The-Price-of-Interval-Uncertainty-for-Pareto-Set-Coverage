# Kaggle execution package

The scripts are authoritative; no notebook is required. Copy the entire `Code/` directory to a Kaggle working directory, preserving `src/`, `experiments/`, `validation/`, `kaggle/`, and `requirements.txt`. Use a Python environment with NumPy and Matplotlib; PyTorch is optional for batched CUDA midpoint runs. Do not use TPU. No dataset is bundled. Stage E requires a user-supplied chronological `.npz` with arrays described in `experiments/streaming_classification.py`.

From the copied `Code/` directory, run each stage deliberately:

```bash
python kaggle/run_all.py --stage A
python kaggle/run_all.py --stage B
python kaggle/run_all.py --stage C --device auto
python kaggle/run_all.py --stage D --device auto
python kaggle/run_all.py --stage E --data /kaggle/input/YOUR_DATA/stream.npz
python kaggle/run_all.py --stage F --device auto
python kaggle/run_all.py --stage G --device auto
python kaggle/run_all.py --stage H
```

**Gate:** Stage B runs a tiny smoke case and, if CUDA is present, checks float64 CUDA batched midpoint results against the CPU NumPy implementation. Do not continue to C–H if parity or theorem checks fail. Stage C executes one representative seed of each synthetic experiment. Stage D performs the full seed sweeps except ablations and horizon scaling, which are deferred to F and G. Stage E is exploratory and cannot validate latent containment without a calibration study. Stage H collects all designated per-seed results and makes tables/figures without selecting favorable seeds.

`kaggle/configs/full.json` specifies 10 seeds, all sweep values, and output root. `kaggle/run_synthetic.py` also supports `--experiments`, `--seed-mode one|full`, `--device cpu|auto|cuda`, and `--max-jobs` for narrow checks. Independent seed/configuration/preference runs get a SHA-256-derived ID and an atomic per-run JSON checkpoint. A valid success checkpoint is skipped on resume. The implementation version is part of each checkpoint ID; bump it when experimental behavior changes. Failures are recorded under `results/logs/` and re-raised. Stage H refuses to aggregate incomplete seed groups. GPU acceleration batches only independent seeds of the same midpoint setting; the recurrence from action to feedback to next action remains sequential. The explicit upper/lower/static baselines and Pareto-grid experiment run on CPU because their small control-heavy operations are not expected to benefit from GPU.

The full matrix includes zero width, width and drift sweeps, horizon scaling, joint width×drift, objective count, decision dimension, a one-dimensional moving front, and endpoint/static ablations. It uses five reproducible preferences per setting (structured 2-objective grid or fixed simplex samples in higher dimensions). Each checkpoint keeps per-seed metrics, preference, timing, device, package versions, timestamp and Git hash when available. `results/tables/` stores per-seed, aggregate and worst-preference CSVs; confidence intervals use `mean ± 1.96·sd/√n` as descriptive normal approximations. Figures are aggregate diagnostics, not proof of a rate or claim of superiority.

The optional CUDA path was not executed on the local laptop. The Kaggle version 2 commit passed Stage B CPU/CUDA parity on both Tesla T4 devices and completed the full synthetic suite; the downloaded `D:/JCAM/result_jcam.zip` records 2,460 per-seed synthetic runs in 246 ten-seed groups. EXP-10 was run separately in `JCAM_EXP10_Bank_Marketing.ipynb` using a private UCI Bank Marketing stream dataset; see `Code/EXP10_REAL_DATA_REPORT.md`. The real-data application has no theorem certificate until population interval containment and feedback assumptions are established.

EXP-10 follow-ups are stored as Kaggle versions 3–5 of the same notebook. Version 3 audits chronological forecast coverage and matched preference learners. Version 4 computes uniform population-band widths under APP-01's conditional within-batch IID model and includes the proof in `result_jcam_exp10_population_bands.zip`. The UCI chronology does not establish that model, and the calculated bands are too wide for a useful real-data theorem claim.

Version 5 adds a controlled IID-with-replacement replay from each fixed chronological window, with known finite-window population objectives and ten seeds. Its `result_jcam_exp10_controlled_resampling.zip` includes the proof, width calculation, per-round errors, and summary. The controlled check confirms played-action containment but exposes the conservatism of the uniform bands; it does not validate within-batch IID for the original UCI observations.

When Kaggle exposes two T4 devices, the importable notebook launches two independent workers for stages D, F and G. Worker 0 takes seeds with remainder 0 modulo 2 and sees physical GPU 0; worker 1 takes remainder 1 and sees physical GPU 1. No online time trajectory is split. Both workers write different checkpoint IDs, and Stage H waits for both and checks that every configured group has all seeds. Stage B checks CPU/CUDA parity on each visible device before these workers start.
