"""Run the minimal KGRL example: synthetic inputs -> PPO -> paths -> metrics."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    started = time.perf_counter()
    import numpy as np
    import pandas as pd
    import torch
    from kgrl.core.device import set_device
    from kgrl.core.seed import set_seed
    from scripts.generate_sample_data import generate

    torch.set_num_threads(1)
    set_device("cpu")
    from kgrl.demo import Demo

    config = json.loads((ROOT / "configs/demo.json").read_text(encoding="utf-8"))
    if config["training"]["ppo_batch_size"] < 2 or config["training"]["num_episodes"] < 2:
        raise ValueError("Use at least two episodes and PPO batch size >= 2")
    set_seed(config["seed"])
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "verification.json").write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")
    timings = {}

    def stage(name, fn):
        print(name + " ...", flush=True)
        start = time.perf_counter()
        result = fn()
        timings[name] = round(time.perf_counter() - start, 3)
        return result

    stage("generate_data", lambda: generate(ROOT))
    demo = Demo(ROOT, config)
    training = stage("train_ppo", lambda: demo.train(out))
    set_seed(config["seed"] + 1)
    predictions = stage("infer", lambda: demo.infer(out))
    evaluation = stage("evaluate", lambda: demo.evaluate(predictions, out))

    losses = pd.read_csv(out / "losses.csv")
    assert len(losses) > 0 and np.isfinite(losses.select_dtypes("number")).all().all(), "Invalid PPO losses"
    assert training["actor_parameter_l1_change"] > 0, "Actor did not learn/update"
    assert set(pd.read_csv(out / "training.csv").stage) == {1, 2}, "Both curriculum stages must run"
    assert set(predictions.Start_Semiology) == set(demo.nodes), "Missing inference starts"
    assert np.isfinite(evaluation.select_dtypes("number")).all().all(), "Nonfinite output"
    assert evaluation[["Sequence", "LCS_F1", "WLCS_F1", "Edge_F1"]].apply(lambda c: c.between(0, 1).all()).all()
    for row in predictions.to_dict("records"):
        path = row["Path"].split(" -> ")
        assert 1 < len(path) <= config["inference"]["max_steps"] + 1
        assert path[0] == row["Start_Semiology"] and len(path) == len(set(path))
        assert len(path) == len(row["SR_Sequence"].split(" -> "))
        assert all(b in {n for _, n in demo.graph[a]} for a, b in zip(path, path[1:])), "Invalid graph edge"
    manifest = json.loads((ROOT / "docs/source_manifest.json").read_text())
    for item in manifest["unchanged_core"]:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"], item["path"]
    elapsed = round(time.perf_counter() - started, 3)
    verification = {"status": "PASS", "scope": "synthetic core-algorithm execution", "seed": config["seed"],
                    **training, "starts": len(demo.nodes), "predicted_paths": len(predictions),
                    "unchanged_core_files": len(manifest["unchanged_core"]), "timings_seconds": timings,
                    "total_seconds": elapsed}
    environment = {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU", "threads": 1,
                   "packages": {n: importlib.metadata.version(n) for n in ["torch", "numpy", "pandas"]}}
    (out / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (out / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(f"PASS: {training['optimizer_updates']} PPO updates, {len(predictions)} paths; {elapsed:.1f} s. Results: {out}")


if __name__ == "__main__":
    main()
