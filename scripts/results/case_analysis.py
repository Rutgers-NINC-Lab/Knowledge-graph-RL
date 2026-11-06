"""Tables 4-5: subgroup statistics and individual paired-reference examples."""
import pandas as pd
from scripts.experiments.common import ROOT, EXP, RESULTS, SEEDS, metrics, require_baselines, write
from scripts.results.common import bootstrap_ci


def run():
    """Tables 4-5: multi-label subgroups and directly paired illustrative reference scores."""
    require_baselines()
    scores = pd.read_csv(EXP / "baseline_evaluations.csv")
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
        write(pd.DataFrame(rows).sort_values("Mean_frequency", ascending=False), RESULTS / filename)
    refs, vocab = metrics.load_reference_rows(str(ROOT / "data/references.csv"), "Path", True, "info", 3.0, 0)
    reference_rows = pd.read_csv(ROOT / "data/references.csv")
    examples = []
    for row in top[top.Seed == SEEDS[0]].head(4).to_dict("records"):
        index = int(reference_rows.index[reference_rows.Start_Semiology == row["Start_Semiology"]][0])
        score = metrics.score_candidate(row["SR_Sequence"].split(" -> "), refs[index], 0.0, vocab, False)
        examples.append({"Start": row["Start_Semiology"], "Semiology_path": row["Path"], "Predicted_SR_path": row["SR_Sequence"],
                         "Paired_reference": reference_rows.iloc[index].Path, "Sequence": score["sequence_score"]})
    write(pd.DataFrame(examples), RESULTS / "table5.csv")


if __name__ == "__main__":
    run()
