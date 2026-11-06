# Knowledge Graph RL

A compact implementation of knowledge graph reinforcement learning for **Uncovering Epileptic Seizure Propagation Using Knowledge Graph-based Reinforcement Learning**.

The code retains the KGRL graph environment, SR reasoning, RotatE and path rewards, GRU actor-critic, PPO, DQN, and behavioral cloning. It includes two-stage curriculum training, reward-ranked path sampling, and sequence evaluation.

## Run

Tested with Python 3.10.21 on Windows using CPU execution. In a Python environment, install the dependencies and run the core example:

```bash
python -m pip install -r requirements.txt
python run_demo.py
```

The example generates synthetic inputs, trains PPO for 24 episodes, reloads the model, decodes paths, and evaluates them. A successful run prints `PASS`. No original KGRL checkout or pre-existing data files are required.

To run the baseline comparisons, ablations, diagnostics, and plotting workflows:

```bash
python scripts/run_experiments.py
```

On Windows, `RUN_DEMO.ps1` runs the core example; add `-Experiments` for the full workflow. Use `-Python 'C:\path\to\python.exe'` to select an interpreter explicitly. The earlier `-Report` switch remains an alias.

The experiments can also be run separately from the repository directory. Run the baseline comparison first to create the shared PPO checkpoints:

```bash
python -m scripts.experiments.baselines
python -m scripts.experiments.ablations
python -m scripts.experiments.decoding
python -m scripts.experiments.diagnostics
```

## Code

| Path | Purpose |
|---|---|
| `run_demo.py` | Core training, inference, evaluation, and execution checks |
| `configs/demo.json` | Demo settings |
| `src/kgrl/` | Algorithms, actor-critic, graph environment, SR reasoning, rewards, and metrics |
| `scripts/generate_sample_data.py` | Synthetic input generator |
| `scripts/run_experiments.py` | Run and verify the complete experiment suite |
| `scripts/experiments/baselines.py` | PPO, DQN, BC and non-learning baseline comparisons |
| `scripts/experiments/ablations.py` | Four component ablations |
| `scripts/experiments/decoding.py` | PPO beam-decoding comparison |
| `scripts/experiments/diagnostics.py` | Candidate-budget and path-length sweeps |
| `scripts/results/` | Aggregate experiment outputs, calculate statistics, and draw figures |
| `docs/source_manifest.json` | Source provenance and hashes checked by the core example |

Both entry points create `data/` and `outputs/` locally. Experiment paths, checkpoints, and training logs go to `outputs/experiments/`; final tables, figure data, and six PNG/PDF plots go to `outputs/results/`. These generated files, checkpoints, caches, and the journal reproducibility report are excluded from version control.

## Figure and table code

Paths below are relative to `scripts/`. Result modules read generated experiment outputs and do not retrain the models.

| Manuscript item | Experiment code | Result code |
|---|---|---|
| Figures 3-5 | Input-data analysis | `results/knowledge_analysis.py` |
| Figure 7 | `experiments/diagnostics.py` | `results/diagnostics.py` |
| Figure 8 | `experiments/baselines.py` supplies candidates | `results/diagnostics.py` |
| Figure 9B and network data | `experiments/baselines.py` supplies paths | `results/propagation_network.py` |
| Table 1 | `experiments/baselines.py` | `results/comparison_tables.py` |
| Table 2 | `experiments/ablations.py` | `results/comparison_tables.py` |
| Table 3 | `experiments/decoding.py` | `results/comparison_tables.py` |
| Tables 4(a,b) and 5 | `experiments/baselines.py` supplies paths | `results/case_analysis.py` |

For result-only processing, run modules such as `python -m scripts.results.comparison_tables` or `python -m scripts.results.diagnostics` after the corresponding experiments have completed.

All inputs and results in these examples are synthetic. The workflows verify execution rather than recover the manuscript's clinical estimates. Figure 9 includes network data and the panel-B matrix; anatomical panel A is not rendered. Figures 1, 2, and 6 are diagrams and are not generated.

Twelve core modules are unchanged from KGRL. Sampling and metric definitions preserve their original calculations; behavioral cloning uses the original temporal-soft-target loss with the shared actor interface. No clinical records, trained embeddings, or pretrained models are distributed.

## License

MIT; see `LICENSE`.
