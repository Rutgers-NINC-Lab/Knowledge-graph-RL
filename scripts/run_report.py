"""Generate the data behind all computational report entries on synthetic inputs."""
import copy
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from kgrl.core.device import set_device
set_device("cpu")
torch.set_num_threads(1)
from kgrl.core.seed import set_seed
from kgrl.demo import Demo
from kgrl import metrics
from generate_sample_data import generate

OUT = ROOT / "outputs/report"
SEEDS = [42, 43]
SCORES = ["Sequence@1", "Sequence@3", "Sequence@5"]


def write(frame, name):
    frame.to_csv(OUT / name, index=False)
    return frame


def summarize(frame, method, seed):
    row = {"Method": method, "Seed": seed}
    for k in [1, 3, 5]:
        selected = frame[frame.Rank <= k]
        row[f"Sequence@{k}"] = selected.groupby("Start_Semiology").Sequence.mean().mean()
    if method == "Greedy Decoding":
        row["Sequence@3"] = row["Sequence@5"] = np.nan
    return row


def mean_sd(frame):
    means = frame.groupby("Method", sort=False)[SCORES].mean()
    sds = frame.groupby("Method", sort=False)[SCORES].std().fillna(0.0)
    for col in SCORES:
        means[col + "_SD"] = sds[col].where(means[col].notna())
    return means.reset_index()


def beam_decode(demo, directory, policy=True):
    """Small beam search under the shared environment; rank by accumulated log probability."""
    checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    actor = demo.actor()
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    rows = []
    for start in demo.nodes:
        env = demo.environment(max_steps=3)
        for key, value in checkpoint["env_state"].items():
            setattr(env, key, value.copy())
        env.reset(start)
        beam = [(env, actor.init_hidden(), 0.0, 0.0, False)]
        for _ in range(3):
            expanded = []
            for state, hidden, reward, logprob, done in beam:
                actions = state.get_candidate_actions()
                if done or not actions:
                    expanded.append((state, hidden, reward, logprob, True))
                    continue
                with torch.no_grad():
                    logits, _, next_hidden, _ = actor(demo.tensors[start], demo.tensors[state.current_node_str],
                                                    hidden, actions, demo.tensors)
                    if not policy:
                        logits = torch.tensor([a[4] for a in actions])
                    log_probs = logits.log_softmax(0)
                for i, action in enumerate(actions):
                    child = copy.deepcopy(state)
                    _, delta, child_done, _ = child.step(action, None)
                    expanded.append((child, next_hidden, reward + delta, logprob + float(log_probs[i]), child_done))
            beam = sorted(expanded, key=lambda x: (-x[3], " -> ".join(x[0].path_history)))[:5]
        seen = set()
        for state, _, reward, logprob, _ in beam:
            sr = " -> ".join(state.sr_history)
            if sr not in seen:
                seen.add(sr)
                rows.append({"Rank": len(seen), "Start_Semiology": start, "Path": " -> ".join(state.path_history),
                             "SR_Sequence": sr, "Total_Reward": reward, "LogProb": logprob})
    return pd.DataFrame(rows)


def model_comparisons(config):
    """Tables 1-3: shared core training, baselines, four ablations and PPO beam decoding."""
    main, ablations, decoders, checks, evaluations, candidates = [], [], [], [], [], []
    variants = ["PPO", "DQN", "Behavioral Cloning", "w/o temporal reward", "w/o path regularization",
                "w/o SR connectivity", "w/o curriculum training"]
    for seed in SEEDS:
        for variant in variants:
            cfg = copy.deepcopy(config)
            cfg["seed"] = seed
            if variant == "w/o temporal reward":
                cfg["environment"]["temporal_scale"] = 0.0
            if variant == "w/o path regularization":
                cfg["environment"].update(penalty_scale=0.0, anti_hacking_penalty_enabled=False)
                cfg["training"]["anti_hub_loss_enabled"] = False
            if variant == "w/o SR connectivity":
                cfg["environment"].update(candidate_mode="graph_neighbors", sr_belief_enabled=False,
                                          reachability_filter_enabled=False)
            algorithm = {"DQN": "dqn", "Behavioral Cloning": "bc"}.get(variant, "ppo")
            set_seed(seed)
            demo = Demo(ROOT, cfg)
            with tempfile.TemporaryDirectory(prefix="iise_report_") as temporary:
                directory = OUT / "models" / f"ppo_seed{seed}" if variant == "PPO" else Path(temporary)
                directory.mkdir(parents=True, exist_ok=True)
                result = demo.train(directory, algorithm, curriculum=variant != "w/o curriculum training")
                losses = pd.read_csv(directory / "losses.csv")
                assert len(losses) and np.isfinite(losses.total_loss).all() and result["actor_parameter_l1_change"] > 0
                tag = variant.replace("/", "").replace(" ", "_").lower() + f"_seed{seed}"
                shutil.copy2(directory / "losses.csv", OUT / "logs" / f"{tag}.csv")
                checks.append({"Method": variant, "Seed": seed, **result})
                set_seed(seed + 1)
                prediction = demo.infer(directory)
                scored = demo.evaluate(prediction, directory)
                scored["Method"], scored["Seed"] = variant, seed
                evaluations.append(scored)
                summary = summarize(scored, variant, seed)
                (main if variant in variants[:3] else ablations).append(summary)
                if variant != "PPO":
                    continue
                ablations.append(summary)
                decoders.append(summary)
                pool = demo.evaluate(demo.last_candidates.drop_duplicates(["Start_Semiology", "SR_Sequence"]), directory, write=False)
                pool["Method"], pool["Seed"] = "PPO", seed
                candidates.append(pool)
                for mode, label in [("score", "Score Rerank"), ("random", "MC Rollout"), ("greedy", "Greedy Decoding"),
                                    ("beam_rank", "Beam Search"), ("beam_policy", "PPO beam search")]:
                    if seed != SEEDS[0] and mode in {"greedy", "beam_rank", "beam_policy"}:
                        continue
                    set_seed(seed + 1)
                    prediction = beam_decode(demo, directory, policy=mode == "beam_policy") if mode.startswith("beam") else demo.infer(directory, mode=mode, write=False)
                    scored = demo.evaluate(prediction, directory, write=False)
                    scored["Method"], scored["Seed"] = label, seed
                    evaluations.append(scored)
                    (decoders if mode == "beam_policy" else main).append(summarize(scored, label, seed))
                    if mode == "score":
                        pool = demo.evaluate(demo.last_candidates.drop_duplicates(["Start_Semiology", "SR_Sequence"]), directory, write=False)
                        pool["Method"], pool["Seed"] = label, seed
                        candidates.append(pool)
            print(f"Completed {variant}, seed {seed}", flush=True)
    write(pd.DataFrame(checks), "training_checks.csv")
    write(pd.concat(evaluations, ignore_index=True), "evaluations.csv")
    write(pd.concat(candidates, ignore_index=True), "candidate_scores.csv")
    for number, rows in [(1, main), (2, ablations), (3, decoders)]:
        data = write(pd.DataFrame(rows), f"table{number}_seeds.csv")
        table = mean_sd(data)
        if number == 2:
            baseline = float(table.loc[table.Method == "PPO", "Sequence@5"].iloc[0])
            table["Delta@5"] = table["Sequence@5"] - baseline
        write(table, f"table{number}.csv")


def knowledge_and_agreement(config):
    """Figures 3-5: associations, connectivity and agreement scores from the input files."""
    demo = Demo(ROOT, config)
    regions = demo.sample["regions"]
    association = pd.DataFrame([[int(sr in demo.kb.get_srs_for_semiology(n)) for sr in regions] for n in demo.nodes],
                               index=demo.nodes, columns=regions)
    association.to_csv(OUT / "figure3_association.csv", index_label="Semiology")
    edges = sorted({tuple(sorted((a, b))) for a in regions for b in demo.kb.get_connected_srs(a) if a != b})
    write(pd.DataFrame(edges, columns=["Source", "Target"]), "figure4_edges.csv")
    pairs = pd.read_csv(ROOT / "data/expert_pairs.csv")
    values = []
    for row in pairs.to_dict("records"):
        a, b = row["Expert_A"].split(" -> "), row["Expert_B"].split(" -> ")
        values.append(metrics.lcs_length(a, [{n} for n in b]) / min(len(a), len(b)))
    pairs["Agreement"] = values
    write(pairs, "figure5_agreement.csv")
    lo, hi = bootstrap_ci(np.array(values))
    write(pd.DataFrame([{"N": len(values), "Mean": np.mean(values), "Median": np.median(values), "CI_low": lo, "CI_high": hi}]), "figure5_summary.csv")


def bootstrap_ci(values):
    rng = np.random.default_rng(20261106)
    means = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def inference_diagnostics(config):
    """Figure 7: smaller nested candidate budgets and horizon sweeps using the saved PPO models."""
    budgets, horizons = [], []
    for seed in SEEDS:
        cfg = copy.deepcopy(config)
        cfg["seed"] = seed
        demo = Demo(ROOT, cfg)
        directory = OUT / "models" / f"ppo_seed{seed}"
        set_seed(seed + 1)
        demo.infer(directory, num_samples=64, starts=demo.nodes[:4], write=False)
        for start, pool in demo.last_candidates.groupby("Start_Semiology"):
            oracle = pool.Total_Reward.max()
            for n in [8, 16, 32, 64]:
                budgets.append({"Seed": seed, "Start": start, "Budget": n,
                                "Reward_gap": oracle - pool.iloc[:n].Total_Reward.max()})
        for length in [3, 4, 5, 6]:
            set_seed(seed + 1)
            pred = demo.infer(directory, max_steps=length-1, num_samples=32, starts=demo.nodes[:4], write=False)
            score = demo.evaluate(pred, directory, write=False)
            horizons.append({"Seed": seed, "Length": length, "Sequence@5": summarize(score, "PPO", seed)["Sequence@5"]})
    write(pd.DataFrame(budgets), "figure7_budget.csv")
    write(pd.DataFrame(horizons), "figure7_horizon.csv")


def reward_alignment():
    """Figure 8: within-start Pearson correlations; undefined groups remain missing."""
    pool = pd.read_csv(OUT / "candidate_scores.csv")
    pool["Final"] = 0.45 * pool.LCS_F1 + 0.45 * pool.WLCS_F1 + 0.1 * pool.Edge_F1
    rows = []
    for (method, seed, start), group in pool.groupby(["Method", "Seed", "Start_Semiology"]):
        for metric in ["Sequence", "LCS_F1", "WLCS_F1", "Final"]:
            r = group.Total_Reward.corr(group[metric]) if len(group) > 2 and group.Total_Reward.std() > 0 and group[metric].std() > 0 else np.nan
            rows.append({"Method": method, "Seed": seed, "Start": start, "Metric": metric, "Pearson_r": r, "N": len(group)})
    write(pd.DataFrame(rows), "figure8_correlations.csv")


def subgroup_and_examples():
    """Tables 4-5: multi-label subgroups and directly paired illustrative reference scores."""
    scores = pd.read_csv(OUT / "evaluations.csv")
    top = scores[(scores.Method == "PPO") & (scores.Rank == 1)].merge(pd.read_csv(ROOT / "data/subgroups.csv"), on="Start_Semiology", validate="many_to_one")
    regional = top.assign(General_Regions=top.General_Regions.str.split("|")).explode("General_Regions")
    for label, frame, filename in [("General_Regions", regional, "table4a.csv"), ("Clinical_Type", top, "table4b.csv")]:
        rows = []
        for group, values in frame.groupby(label):
            seed_means = values.groupby("Seed").Sequence.mean()
            subject_means = values.groupby("Start_Semiology").Sequence.mean().to_numpy()
            lo, hi = bootstrap_ci(subject_means)
            rows.append({"Group": group, "Semiologies": len(subject_means), "Sequence@1": seed_means.mean(),
                         "Across_seed_SD": seed_means.std(), "Bootstrap_CI_low": lo, "Bootstrap_CI_high": hi,
                         "Mean_frequency": values.Frequency.mean()})
        write(pd.DataFrame(rows).sort_values("Mean_frequency", ascending=False), filename)
    refs, vocab = metrics.load_reference_rows(str(ROOT / "data/references.csv"), "Path", True, "info", 3.0, 0)
    reference_rows = pd.read_csv(ROOT / "data/references.csv")
    examples = []
    for row in top[top.Seed == SEEDS[0]].head(4).to_dict("records"):
        index = int(reference_rows.index[reference_rows.Start_Semiology == row["Start_Semiology"]][0])
        score = metrics.score_candidate(row["SR_Sequence"].split(" -> "), refs[index], 0.0, vocab, False)
        examples.append({"Start": row["Start_Semiology"], "Semiology_path": row["Path"], "Predicted_SR_path": row["SR_Sequence"],
                         "Paired_reference": reference_rows.iloc[index].Path, "Sequence": score["sequence_score"]})
    write(pd.DataFrame(examples), "table5.csv")


def network_aggregation():
    """Figure 9: node/edge data for panel A and a normalized transition matrix for panel B."""
    scores = pd.read_csv(OUT / "evaluations.csv")
    paths = scores[(scores.Method == "PPO") & (scores.Seed == SEEDS[0]) & (scores.Rank <= 5)]
    edges = []
    for sequence in paths.SR_Sequence:
        nodes = sequence.split(" -> ")
        edges.extend(zip(nodes, nodes[1:]))
    frame = pd.DataFrame(edges, columns=["Source", "Target"])
    edge_counts = frame.value_counts().rename("Count").reset_index()
    write(edge_counts, "figure9_edges.csv")
    matrix = edge_counts.pivot(index="Source", columns="Target", values="Count").fillna(0)
    regions = json.loads((ROOT / "data/sample.json").read_text())["regions"]
    matrix = matrix.reindex(index=regions, columns=regions, fill_value=0)
    matrix.to_csv(OUT / "figure9_counts.csv")
    (matrix / max(float(matrix.to_numpy().max()), 1.0)).to_csv(OUT / "figure9_matrix.csv")
    degree = matrix.sum(axis=0).add(matrix.sum(axis=1), fill_value=0)
    write(degree.rename("Path_degree").rename_axis("Region").reset_index(), "figure9_nodes.csv")


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "verification.json").write_text('{"status": "RUNNING"}')
    config = json.loads((ROOT / "configs/demo.json").read_text())
    timings = {}
    def timed(name, function):
        begin = time.perf_counter()
        function()
        timings[name] = round(time.perf_counter() - begin, 3)
        print(f"{name}: {timings[name]:.3f} s", flush=True)
    timed("generate", lambda: generate(ROOT))
    timed("tables1_3_shared", lambda: model_comparisons(config))
    timed("figures3_5_data", lambda: knowledge_and_agreement(config))
    timed("figure7_data", lambda: inference_diagnostics(config))
    timed("figure8_data", reward_alignment)
    timed("tables4_5_data", subgroup_and_examples)
    timed("figure9_data", network_aggregation)
    from plot_report import plot_all
    timed("plots", lambda: plot_all(OUT))
    expected = [f"table{i}.csv" for i in [1, 2, 3, 5]] + ["table4a.csv", "table4b.csv"]
    expected += [f"figure{i}.png" for i in [3, 4, 5, 7, 8, 9]]
    assert all((OUT / name).is_file() and (OUT / name).stat().st_size for name in expected)
    training = pd.read_csv(OUT / "training_checks.csv")
    assert len(training) == 14 and (training.optimizer_updates > 0).all()
    evaluation = pd.read_csv(OUT / "evaluations.csv")
    assert np.isfinite(evaluation[["Sequence", "LCS_F1", "WLCS_F1", "Edge_F1", "Total_Reward"]]).all().all()
    for path in evaluation.Path:
        nodes = path.split(" -> ")
        assert len(nodes) == len(set(nodes))
    evidence = {"status": "PASS", "scope": "synthetic workflow for computational figures and tables",
                "seeds": SEEDS, "training_jobs": len(training), "evaluated_paths": len(evaluation),
                "timings_seconds": timings, "total_seconds": round(time.perf_counter()-started, 3), "outputs": expected}
    (OUT / "verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    environment = {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU", "threads": 1,
                   "packages": {n: importlib.metadata.version(n) for n in ["torch", "numpy", "pandas", "matplotlib"]}}
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(f"PASS report workflow: {evidence['total_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
