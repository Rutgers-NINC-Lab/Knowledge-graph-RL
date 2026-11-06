"""Figures 7-8: inference plots and reward-metric correlation results."""
import numpy as np
import pandas as pd
from scripts.experiments.common import EXP, RESULTS, require_baselines, write
from scripts.results.common import plt, save


def plot_inference():
    out = EXP
    budgets = pd.read_csv(out / "figure7_budget.csv").groupby(["Seed", "Budget"]).Reward_gap.median().groupby("Budget").agg(["mean", "std"])
    horizon = pd.read_csv(out / "figure7_horizon.csv").groupby("Length")["Sequence@5"].agg(["mean", "std"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, frame, x, y in [(axes[0], budgets, "Candidate budget", "Median reward gap"), (axes[1], horizon, "Path length (nodes)", "Sequence@5")]:
        ax.errorbar(frame.index, frame["mean"], yerr=frame["std"], marker="o", capsize=3, color="#2375a9")
        ax.set(xlabel=x, ylabel=y)
        ax.set_xticks(frame.index)
    save(fig, 7)



def reward_alignment():
    """Figure 8: within-start Pearson correlations; undefined groups remain missing."""
    require_baselines()
    pool = pd.read_csv(EXP / "candidate_scores.csv")
    pool["Final"] = 0.45 * pool.LCS_F1 + 0.45 * pool.WLCS_F1 + 0.1 * pool.Edge_F1
    rows = []
    for (method, seed, start), group in pool.groupby(["Method", "Seed", "Start_Semiology"]):
        for metric in ["Sequence", "LCS_F1", "WLCS_F1", "Final"]:
            r = group.Total_Reward.corr(group[metric]) if len(group) > 2 and group.Total_Reward.std() > 0 and group[metric].std() > 0 else np.nan
            rows.append({"Method": method, "Seed": seed, "Start": start, "Metric": metric, "Pearson_r": r, "N": len(group)})
    write(pd.DataFrame(rows), RESULTS / "figure8_correlations.csv")
    plot_reward()


def plot_reward():
    out = RESULTS
    correlations = pd.read_csv(out / "figure8_correlations.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, method in zip(axes, ["PPO", "Score Rerank"]):
        frame = correlations[correlations.Method == method].groupby("Metric").Pearson_r.mean()
        ax.bar(frame.index, frame.values, color="#2375a9")
        ax.axhline(0, color="0.3", lw=.7)
        ax.set(title=method, ylabel="Mean within-start Pearson r", ylim=(-1, 1))
        ax.tick_params(axis="x", rotation=30)
    save(fig, 8)



if __name__ == "__main__":
    plot_inference()
    reward_alignment()
