"""Shared plotting style and bootstrap helper."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.experiments.common import RESULTS

plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})


def bootstrap_ci(values):
    rng = np.random.default_rng(20261106)
    means = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def save(fig, number):
    out = RESULTS
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
