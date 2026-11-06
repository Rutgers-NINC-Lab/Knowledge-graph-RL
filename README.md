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
python scripts/run_report.py
```

On Windows, `RUN_DEMO.ps1` runs the core example; add `-Report` for the full experiment workflow. Use `-Python 'C:\path\to\python.exe'` to select an interpreter explicitly.

## Code

| Path | Purpose |
|---|---|
| `run_demo.py` | Core training, inference, evaluation, and execution checks |
| `configs/demo.json` | Demo settings |
| `src/kgrl/` | Algorithms, actor-critic, graph environment, SR reasoning, rewards, and metrics |
| `scripts/generate_sample_data.py` | Synthetic input generator |
| `scripts/run_report.py` | Shared experiments for Figures 3-5, 7-9 and Tables 1-5 |
| `scripts/plot_report.py` | Plot the exported figure data |
| `docs/source_manifest.json` | Source provenance and hashes checked by the core example |

Both entry points create `data/` and `outputs/` locally. The full workflow writes CSVs and six PNG/PDF plots to `outputs/report/`. These generated files, checkpoints, caches, and the journal reproducibility report are excluded from version control.

All inputs and results in these examples are synthetic. The workflows verify execution rather than recover the manuscript's clinical estimates. Figure 9 includes network data and the panel-B matrix; anatomical panel A is not rendered. Figures 1, 2, and 6 are diagrams and are not generated.

Twelve core modules are unchanged from KGRL. Sampling and metric definitions preserve their original calculations; behavioral cloning uses the original temporal-soft-target loss with the shared actor interface. No clinical records, trained embeddings, or pretrained models are distributed.

## License

MIT; see `LICENSE`.
