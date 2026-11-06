"""Table 2 experiments: temporal reward, regularization, SR connectivity and curriculum."""
import copy
import pandas as pd
from scripts.experiments.common import EXP, SEEDS, fit_model, prepare, require_baselines, write


def run(config):
    require_baselines()
    evaluations, checks = [], []
    for seed in SEEDS:
        for method in ["w/o temporal reward", "w/o path regularization", "w/o SR connectivity", "w/o curriculum training"]:
            cfg = copy.deepcopy(config)
            if method == "w/o temporal reward":
                cfg["environment"]["temporal_scale"] = 0.0
            if method == "w/o path regularization":
                cfg["environment"].update(penalty_scale=0.0, anti_hacking_penalty_enabled=False)
                cfg["training"]["anti_hub_loss_enabled"] = False
            if method == "w/o SR connectivity":
                cfg["environment"].update(candidate_mode="graph_neighbors", sr_belief_enabled=False, reachability_filter_enabled=False)
            with fit_model(cfg, seed, method, curriculum=method != "w/o curriculum training") as (_, _, scored, check):
                evaluations.append(scored)
                checks.append(check)
    write(pd.concat(evaluations, ignore_index=True), EXP / "ablation_evaluations.csv")
    write(pd.DataFrame(checks), EXP / "ablation_training.csv")


if __name__ == "__main__":
    run(prepare())
    from scripts.results.comparison_tables import export_table
    export_table(2)
