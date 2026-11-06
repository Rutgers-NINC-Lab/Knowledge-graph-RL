"""Figure 7 experiment: sampling budget and path-length sweeps."""
import copy
import pandas as pd
from scripts.experiments.common import ROOT, EXP, SEEDS, Demo, prepare, require_baselines, set_seed, write
from scripts.results.comparison_tables import summarize


def run(config):
    """Figure 7: smaller nested candidate budgets and horizon sweeps using the saved PPO models."""
    require_baselines()
    budgets, horizons = [], []
    for seed in SEEDS:
        cfg = copy.deepcopy(config)
        cfg["seed"] = seed
        demo = Demo(ROOT, cfg)
        directory = EXP / "models" / f"ppo_seed{seed}"
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
    write(pd.DataFrame(budgets), EXP / "figure7_budget.csv")
    write(pd.DataFrame(horizons), EXP / "figure7_horizon.csv")


if __name__ == "__main__":
    run(prepare())
    from scripts.results.diagnostics import plot_inference
    plot_inference()
