"""Figure 9: aggregate predicted paths and draw the panel-B matrix."""
import json
import pandas as pd
from scripts.experiments.common import ROOT, EXP, RESULTS, SEEDS, require_baselines, write
from scripts.results.common import heatmap, plt, save


def run():
    """Figure 9: node/edge data for panel A and a normalized transition matrix for panel B."""
    require_baselines()
    scores = pd.read_csv(EXP / "baseline_evaluations.csv")
    paths = scores[(scores.Method == "PPO") & (scores.Seed == SEEDS[0]) & (scores.Rank <= 5)]
    edges = []
    for sequence in paths.SR_Sequence:
        nodes = sequence.split(" -> ")
        edges.extend(zip(nodes, nodes[1:]))
    frame = pd.DataFrame(edges, columns=["Source", "Target"])
    edge_counts = frame.value_counts().rename("Count").reset_index()
    write(edge_counts, RESULTS / "figure9_edges.csv")
    matrix = edge_counts.pivot(index="Source", columns="Target", values="Count").fillna(0)
    regions = json.loads((ROOT / "data/sample.json").read_text())["regions"]
    matrix = matrix.reindex(index=regions, columns=regions, fill_value=0)
    matrix.to_csv(RESULTS / "figure9_counts.csv")
    (matrix / max(float(matrix.to_numpy().max()), 1.0)).to_csv(RESULTS / "figure9_matrix.csv")
    degree = matrix.sum(axis=0).add(matrix.sum(axis=1), fill_value=0)
    write(degree.rename("Path_degree").rename_axis("Region").reset_index(), RESULTS / "figure9_nodes.csv")
    plot()


def plot():
    out = RESULTS
    fig, ax = plt.subplots(figsize=(7, 5))
    heatmap(ax, pd.read_csv(out / "figure9_matrix.csv", index_col=0), "Count / largest count")
    ax.set_ylabel("Source region")
    ax.set_title("Panel B: path aggregation matrix")
    save(fig, 9)


if __name__ == "__main__":
    run()
