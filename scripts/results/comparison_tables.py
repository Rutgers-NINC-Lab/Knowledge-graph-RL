"""Aggregate experiment paths into the means and seed SDs in Tables 1-3."""
import numpy as np
import pandas as pd
from scripts.experiments.common import EXP, RESULTS, require_baselines, write

SCORES = ["Sequence@1", "Sequence@3", "Sequence@5"]


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


def export_table(number):
    require_baselines()
    RESULTS.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(EXP / "baseline_evaluations.csv")
    if number == 1:
        scores = baseline
    else:
        filename = {2: "ablation_evaluations.csv", 3: "decoding_evaluations.csv"}[number]
        scores = pd.concat([baseline[baseline.Method == "PPO"], pd.read_csv(EXP / filename)], ignore_index=True)
    rows = [summarize(group, method, seed) for (method, seed), group in scores.groupby(["Method", "Seed"], sort=False)]
    data = write(pd.DataFrame(rows), RESULTS / f"table{number}_seeds.csv")
    table = mean_sd(data)
    if number == 2:
        baseline_score = float(table.loc[table.Method == "PPO", "Sequence@5"].iloc[0])
        table["Delta@5"] = table["Sequence@5"] - baseline_score
    write(table, RESULTS / f"table{number}.csv")


if __name__ == "__main__":
    for number in [1, 2, 3]:
        export_table(number)
