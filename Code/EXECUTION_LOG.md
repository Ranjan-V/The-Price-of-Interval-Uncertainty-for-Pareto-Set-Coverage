# Execution log

All times are local (Asia/Calcutta), 2026-10-01 unless stated otherwise. Commands are summarized; detailed machine-readable outputs live under `results/` after execution.

| Stage | Command or action | Purpose | Environment | Outcome / fix |
|---|---|---|---|---|
| Phase 0–1 | Read `RESEARCH_STATUS.md`, relevant Math/Theory files, source, configs, drivers and tests; static audit | Establish exact theorem/code scope | Laptop, read-only | Completed. Classification feedback and containment differ from synthetic theorem model; documented in `Math/implementation_consistency_audit.md`. |
| Phase 2 | `py -V` and Python package inspection | Inspect default Python | Windows 11, Python 3.12.0, 16 logical CPUs, ~15.6 GiB RAM | Default interpreter lacks NumPy, SciPy, pandas, matplotlib, pytest and torch. |
| Phase 2 | Bundled Python package inspection | Identify reusable dependencies | Bundled Python 3.12.14 | NumPy/pandas present; pytest/matplotlib/torch/SciPy absent. No GPU assumed or used. |
| Phase 2 | `py -m venv .venv`; `.venv\\Scripts\\python.exe -m pip install "numpy>=1.26" "pytest>=8.0" "matplotlib>=3.8"` | Isolated CPU test and plotting environment | Project-local Python 3.12 virtual environment | Success. Installed NumPy 2.5.3 (core array dependency), pytest 9.1.1 (existing tests), Matplotlib 3.11.2 (required smoke figures), plus their transitive dependencies. No SciPy/PyTorch installed. |

| Phase 4 | `python -m pytest tests/test_intervals_pareto.py -q` | Interval and Pareto tests | CPU venv | 2 passed. |
| Phase 4 | `python -m pytest tests/test_theory_consistency.py -q` | Variation/regret/synthetic tests | CPU venv | 4 passed. |
| Phase 4 | `python -m pytest tests/test_application_and_lower_bound.py tests/test_parallel_front.py -q` | Application and integration tests | CPU venv | 4 passed. |
| Phase 4 | `python -m pytest tests -q` | Full suite | CPU venv | 10 passed. |
| Phase 5 | `python validation/smoke.py --output results/smoke/run1` | First tiny end-to-end run | CPU venv | **Failed** at JSON serialization after computation: NumPy boolean flags were not serializable. Fixed by explicit `bool()` conversion in scalar and front experiment summaries. |
| Phase 5 | Same smoke command, rerun `run1` and `run2` | End-to-end and deterministic reproduction | CPU venv | Both succeeded; run bodies 0.298 s and 0.244 s. Summary JSON and rounds CSV hashes matched exactly. Diagnostic PNG inspected. |
| Phase 6 | `python validation/sanity.py` | SANITY-01–06 | CPU venv | All six passed; values in `results/sanity/sanity.json`. |
| Phase 7 | `python validation/theorem_checks.py` | Exact THM-02 components; THM-05 sampled coverage; THM-03 identity | CPU venv | 17/17 checks passed; details in `results/theorem_checks/`. |
| Phase 8 | Added lower-endpoint method and tests; `python -m pytest tests/test_endpoint_baselines.py -q` | Complete relevant internal baselines | CPU venv | 2 passed. |
| Phase 5 | `python experiments/run.py --config configs/smoke.json` | Exercise full small matrix driver and output serialization | CPU venv | Succeeded; result, round and metadata files written under `results/smoke/driver/`. |
| Phase 4 final | `python -m pytest tests -q` | Full suite after baseline changes | CPU venv | 12 passed, 0 failed. |
| Phase 2 | `nvidia-smi -L` | Local GPU inventory only | Laptop | RTX 3050 present. No local GPU computation was used. |
| Phase 14–17 | Added `kaggle/` config, checkpoint, CPU/CUDA batch, staged runner, collector and plotter | Prepare large-scale package | Source only at this point | CUDA path unexecuted; Stage B parity gate required. |
| Package dry run | `python kaggle/run_all.py --stage A/B/C/H --config kaggle/configs/smoke.json --device cpu` | Verify environment, smoke, one representative per experiment and reporting | CPU venv | Initial tiny C completed 9 runs; repeated C skipped 9 after fixing job cap to count considered checkpoints. H generated tables and figures. |
| Package dry run | `python validation/application_smoke.py`; Stage E with toy NPZ | Test application input/output and checkpoint | CPU venv | Toy stream serialized successfully; not a real-data experiment. |
| Package dry run | Stages D/F/G/H with tiny config | Test seed sweeps, ablations, scaling, collection and plotting | CPU venv | 192 current-version per-seed synthetic checkpoints, 96 complete groups, nine figures. No large matrix or GPU run. Legacy pre-version checkpoints were excluded from aggregation after finding and fixing mixed-version collection. |
| Phase 4 final | `python -m pytest tests -q` | Full suite including checkpoint/resume tests | CPU venv | 14 passed, 0 failed. |
| Final validation | Repeated `validation/smoke.py` twice after adding finite/containment checks; compared SHA-256 hashes | Confirm actual action intervals and deterministic outputs | CPU venv | Both passed; final body times 0.407 s and 0.415 s; summary and per-round hashes matched exactly. |
| Final validation | `python -m pytest tests -q`, `validation/sanity.py`, `validation/theorem_checks.py` | Recheck after package/reporting changes | CPU venv | 14 tests passed; six sanity cases passed; 17/17 numerical theorem checks passed. |
| Final validation | `python experiments/run.py --config configs/smoke.json` | Verify small diagnostic matrix after aligning experiment IDs | CPU venv | Succeeded; no large experiment run. |
| Workspace check | `git status --short -- Math Theory Code RESEARCH_STATUS.md` | Determine commit metadata availability without inspecting `Paper/` | Laptop | No `.git` repository found; Kaggle checkpoint `git_commit` will be `null` unless the uploaded copy has Git metadata. |
| Notebook preparation | `python kaggle/build_notebook.py`; `python -m pytest tests -q` | Create importable notebook and recheck tests | CPU venv | Notebook built. Test collection **failed** because this invocation omitted `PYTHONPATH`; classified as configuration, not a source regression. Added `pyproject.toml` with pytest paths and reran the suite successfully (14 passed). |
