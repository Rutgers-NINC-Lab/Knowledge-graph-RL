"""Figures 3-5: knowledge associations, connectivity and annotation agreement."""
import numpy as np
import pandas as pd
from scripts.experiments.common import ROOT, RESULTS, Demo, metrics, prepare, write
from scripts.results.common import bootstrap_ci, heatmap, plt, save


def run(config):
    """Figures 3-5: associations, connectivity and agreement scores from the input files."""
    demo = Demo(ROOT, config)
    regions = demo.sample["regions"]
    association = pd.DataFrame([[int(sr in demo.kb.get_srs_for_semiology(n)) for sr in regions] for n in demo.nodes],
                               index=demo.nodes, columns=regions)
    association.to_csv(RESULTS / "figure3_association.csv", index_label="Semiology")
    edges = sorted({tuple(sorted((a, b))) for a in regions for b in demo.kb.get_connected_srs(a) if a != b})
    write(pd.DataFrame(edges, columns=["Source", "Target"]), RESULTS / "figure4_edges.csv")
    pairs = pd.read_csv(ROOT / "data/expert_pairs.csv")
    values = []
    for row in pairs.to_dict("records"):
        a, b = row["Expert_A"].split(" -> "), row["Expert_B"].split(" -> ")
        values.append(metrics.lcs_length(a, [{n} for n in b]) / min(len(a), len(b)))
    pairs["Agreement"] = values
    write(pairs, RESULTS / "figure5_agreement.csv")
    lo, hi = bootstrap_ci(np.array(values))
    write(pd.DataFrame([{"N": len(values), "Mean": np.mean(values), "Median": np.median(values), "CI_low": lo, "CI_high": hi}]), RESULTS / "figure5_summary.csv")
    plot()


def plot():
    out = RESULTS
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



if __name__ == "__main__":
    run(prepare())
