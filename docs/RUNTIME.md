# GPU environment and full-study runtime reference

The reference hardware recorded in the archived full training runs is one NVIDIA RTX A6000 (48 GB VRAM). The archived environment reports Python 3.9.18 and PyTorch 2.3.1 with a CUDA 11.8 build. The distributed code now targets Python 3.10 and PyTorch 2.11 with a CUDA 12.8 build; the archived timings are reference values, not a new benchmark of that environment.

## Recorded training times

| Completed training run | Archived elapsed time |
|---|---:|
| PPO main-model runs | Approximately 15-24 minutes per run |
| PPO reward-variant runs | Approximately 23-24 minutes per run |
| DQN training | Approximately 4.1 hours per run |

These ranges come from completed April-May 2026 runs with saved final models and W&B elapsed-time summaries. No clinical records or original training configurations are included here.

## Full-workflow planning budget

Allow approximately **1-2 days on a single GPU** for the complete study-scale workflow, including the baseline methods, multiple ablation runs, candidate generation, decoding comparisons, and diagnostic sweeps. This allowance extrapolates from the archived training records and includes time for inference and analysis. It is not a measured end-to-end GPU runtime; the logs do not establish a single wall-clock duration for the entire study. Data preparation, dependency installation, and shared-GPU queueing are excluded.

Graph traversal, data loading, aggregation, and plotting also depend on host resources. Actual elapsed time depends on the dataset, number of seeds, training schedule, inference budget, and available hardware.

`configs/example.json` deliberately uses a different illustrative configuration. Its runtime is separate from this full-study reference. Each execution records the actual device and measured stage times in its local `environment.json` and `verification.json`; those records are not committed to the repository.
