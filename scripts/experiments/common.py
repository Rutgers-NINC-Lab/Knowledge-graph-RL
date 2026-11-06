"""Shared runtime and model fitting for the small synthetic experiments."""
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "outputs/experiments"
RESULTS = ROOT / "outputs/results"
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from kgrl.runtime import configure, load_config
CONFIG = load_config(ROOT)
DEVICE = configure(CONFIG)
SEEDS = CONFIG["experiment_seeds"]

import numpy as np
import pandas as pd
import torch
from kgrl.core.seed import set_seed
from kgrl.demo import Demo
from kgrl import metrics
from scripts.generate_sample_data import generate


def prepare():
    (EXP / "logs").mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT)
    generate(ROOT, config)
    return config


def write(frame, path):
    frame.to_csv(path, index=False)
    return frame


def require_baselines():
    if not (EXP / "baseline_evaluations.csv").is_file():
        raise FileNotFoundError("Run python -m scripts.experiments.baselines first.")


def evaluate(demo, prediction, directory, method, seed):
    scored = demo.evaluate(prediction, directory, write=False)
    scored["Method"], scored["Seed"] = method, seed
    return scored


@contextmanager
def fit_model(config, seed, method, algorithm="ppo", curriculum=True):
    cfg = copy.deepcopy(config)
    cfg["seed"] = seed
    set_seed(seed)
    demo = Demo(ROOT, cfg)
    with tempfile.TemporaryDirectory(prefix="kgrl_experiment_") as temporary:
        directory = EXP / "models" / f"ppo_seed{seed}" if method == "PPO" else Path(temporary)
        directory.mkdir(parents=True, exist_ok=True)
        training = demo.train(directory, algorithm, curriculum=curriculum)
        losses = pd.read_csv(directory / "losses.csv")
        assert len(losses) and np.isfinite(losses.total_loss).all() and training["actor_parameter_l1_change"] > 0
        tag = method.replace("/", "").replace(" ", "_").lower() + f"_seed{seed}"
        shutil.copy2(directory / "losses.csv", EXP / "logs" / f"{tag}.csv")
        set_seed(seed + 1)
        prediction = demo.infer(directory)
        scored = evaluate(demo, prediction, directory, method, seed)
        yield demo, directory, scored, {"Method": method, "Seed": seed, **training}
    print(f"Completed {method}, seed {seed}", flush=True)
