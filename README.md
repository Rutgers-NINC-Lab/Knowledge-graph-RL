# Knowledge Graph RL

Implementation of knowledge graph reinforcement learning for our paper **Uncovering Epileptic Seizure Propagation Using Knowledge Graph-based Reinforcement Learning**.

The code includes the graph environment, SR reasoning, path rewards, GRU actor-critic, PPO, DQN, behavioral cloning, and sequence evaluation.

## Run

Use Python 3.10 with a CUDA-capable GPU. Install the dependencies and run the core workflow:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python run_demo.py
```

To run the baseline comparisons, ablations, diagnostics, and result analysis:

```bash
python scripts/run_experiments.py
```

On Windows, `RUN_DEMO.ps1` runs the core workflow; add `-Experiments` for the experiment suite.

`configs/example.json` contains illustrative parameters and selects CUDA by default. The workflows generate synthetic inputs; the original clinical data, paper hyperparameters, and pretrained models are not included.

The complete study-scale workflow is estimated at 1-2 days on one GPU. This is a planning estimate from archived training runs, not a timing of the supplied example; see [runtime details](docs/RUNTIME.md).

## Code

- `src/kgrl/`: core algorithms, graph environment, rewards, and metrics.
- `scripts/experiments/`: baseline, ablation, decoding, and diagnostic experiments.
- `scripts/results/`: result aggregation, tables, and figures.
- `scripts/generate_sample_data.py`: synthetic input generator.

Generated experiment files go to `outputs/experiments/` and tables and figures to `outputs/results/`. Generated files and the journal reproducibility report are excluded from version control.

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{yang2026uncovering,
  title={Uncovering Epileptic Seizure Propagation Using Knowledge Graph-based Reinforcement Learning},
  author={Yang, Shihao and Wen, Zirui and Rehim, Erafat and Kokkinos, Vasileios and Weil, Alexander and Shih, Yen-Cheng and Liu, Yo-Tsen and Liu, Feng},
  journal={IISE Transactions},
  year={2026},
  publisher={Taylor \& Francis}
}
```

MIT License; see [LICENSE](LICENSE).
