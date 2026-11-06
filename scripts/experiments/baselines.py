"""Table 1 experiments: PPO, DQN, BC and four non-learning baselines."""
import pandas as pd
from scripts.experiments.common import EXP, SEEDS, evaluate, fit_model, prepare, set_seed, write
from scripts.experiments.decoding import beam_decode


def run(config):
    evaluations, candidates, checks = [], [], []
    for seed in SEEDS:
        for method, algorithm in [("PPO", "ppo"), ("DQN", "dqn"), ("Behavioral Cloning", "bc")]:
            with fit_model(config, seed, method, algorithm) as (demo, directory, scored, check):
                evaluations.append(scored)
                checks.append(check)
                if method != "PPO":
                    continue
                candidates.append(evaluate(demo, demo.last_candidates.drop_duplicates(["Start_Semiology", "SR_Sequence"]), directory, method, seed))
                for mode, label in [("score", "Score Rerank"), ("random", "MC Rollout"),
                                    ("greedy", "Greedy Decoding"), ("beam_rank", "Beam Search")]:
                    if seed != SEEDS[0] and mode in {"greedy", "beam_rank"}:
                        continue
                    set_seed(seed + 1)
                    prediction = beam_decode(demo, directory, policy=False) if mode == "beam_rank" else demo.infer(directory, mode=mode, write=False)
                    evaluations.append(evaluate(demo, prediction, directory, label, seed))
                    if mode == "score":
                        candidates.append(evaluate(demo, demo.last_candidates.drop_duplicates(["Start_Semiology", "SR_Sequence"]), directory, label, seed))
    write(pd.concat(evaluations, ignore_index=True), EXP / "baseline_evaluations.csv")
    write(pd.concat(candidates, ignore_index=True), EXP / "candidate_scores.csv")
    write(pd.DataFrame(checks), EXP / "baseline_training.csv")


if __name__ == "__main__":
    run(prepare())
    from scripts.results.comparison_tables import export_table
    export_table(1)
