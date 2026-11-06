"""Six compact plots from the CSVs written by run_report.py; no invented manuscript graphics."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_all(out):
    out = Path(out)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    def save(fig, number):
        fig.suptitle(f"Figure {number} workflow — synthetic example", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        for suffix in ["png", "pdf"]:
            fig.savefig(out / f"figure{number}.{suffix}", dpi=180)
        plt.close(fig)
    def heatmap(ax, frame, label):
        im = ax.imshow(frame.to_numpy(float), aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(frame.columns)), frame.columns, rotation=65, ha="right")
        ax.set_yticks(range(len(frame.index)), frame.index)
        ax.set_xlabel("Target region")
        fig = ax.figure
        fig.colorbar(im, ax=ax, label=label)
    fig, ax = plt.subplots(figsize=(7, 6))
    heatmap(ax, pd.read_csv(out / "figure3_association.csv", index_col=0), "Association indicator")
    save(fig, 3)
    edges = pd.read_csv(out / "figure4_edges.csv")
    nodes = sorted(set(edges.Source) | set(edges.Target))
    angles = np.linspace(0, 2*np.pi, len(nodes), endpoint=False)
    positions = dict(zip(nodes, zip(np.cos(angles), np.sin(angles))))
    fig, ax = plt.subplots(figsize=(6, 5))
    for a, b in edges.itertuples(index=False, name=None):
        ax.plot([positions[a][0], positions[b][0]], [positions[a][1], positions[b][1]], color="0.75", zorder=1)
    for n, (x, y) in positions.items():
        ax.scatter(x, y, s=100, color="#2375a9", zorder=2)
        alignment = "left" if x > .8 else "right" if x < -.8 else "center"
        ax.text(x*1.16, y*1.16, n, ha=alignment, va="center")
    ax.set(xlim=(-1.55, 1.55), ylim=(-1.4, 1.4), aspect="equal")
    ax.axis("off")
    save(fig, 4)
    pairs = pd.read_csv(out / "figure5_agreement.csv")
    summary = pd.read_csv(out / "figure5_summary.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(np.arange(1, len(pairs)+1), pairs.Agreement, color="#2375a9")
    ax.axhline(summary.Mean, color="#bd5b18", label=f"Mean {summary.Mean:.3f}")
    ax.axhspan(summary.CI_low, summary.CI_high, alpha=.12, color="#bd5b18", label="Bootstrap mean 95% CI")
    ax.set(xlabel="Synthetic annotation pair", ylabel="LCS / shorter path length", ylim=(0, 1.05))
    ax.legend(fontsize=8)
    save(fig, 5)
    budgets = pd.read_csv(out / "figure7_budget.csv").groupby(["Seed", "Budget"]).Reward_gap.median().groupby("Budget").agg(["mean", "std"])
    horizon = pd.read_csv(out / "figure7_horizon.csv").groupby("Length")["Sequence@5"].agg(["mean", "std"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, frame, x, y in [(axes[0], budgets, "Candidate budget", "Median reward gap"), (axes[1], horizon, "Path length (nodes)", "Sequence@5")]:
        ax.errorbar(frame.index, frame["mean"], yerr=frame["std"], marker="o", capsize=3, color="#2375a9")
        ax.set(xlabel=x, ylabel=y)
        ax.set_xticks(frame.index)
    save(fig, 7)
    correlations = pd.read_csv(out / "figure8_correlations.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, method in zip(axes, ["PPO", "Score Rerank"]):
        frame = correlations[correlations.Method == method].groupby("Metric").Pearson_r.mean()
        ax.bar(frame.index, frame.values, color="#2375a9")
        ax.axhline(0, color="0.3", lw=.7)
        ax.set(title=method, ylabel="Mean within-start Pearson r", ylim=(-1, 1))
        ax.tick_params(axis="x", rotation=30)
    save(fig, 8)
    fig, ax = plt.subplots(figsize=(7, 5))
    heatmap(ax, pd.read_csv(out / "figure9_matrix.csv", index_col=0), "Count / largest count")
    ax.set_ylabel("Source region")
    ax.set_title("Panel B: path aggregation matrix")
    save(fig, 9)


if __name__ == "__main__":
    plot_all(Path(__file__).resolve().parents[1] / "outputs/report")
