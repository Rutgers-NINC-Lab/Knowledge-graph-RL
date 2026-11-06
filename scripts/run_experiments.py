"""Run the compact experiment suite, then verify its figures and tables."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experiments.common import EXP, RESULTS, SEEDS, np, pd, prepare, write
from scripts.experiments import baselines, ablations, decoding, diagnostics
from scripts.results import comparison_tables, knowledge_analysis, case_analysis
from scripts.results import diagnostics as diagnostic_results
from kgrl.runtime import environment, synchronize


def main():
    started = time.perf_counter()
    config = prepare()
    (RESULTS / "verification.json").write_text('{"status": "RUNNING"}')
    timings = {}

    def timed(name, function):
        synchronize()
        begin = time.perf_counter()
        function()
        synchronize()
        timings[name] = round(time.perf_counter() - begin, 3)
        print(f"{name}: {timings[name]:.3f} s", flush=True)

    for number, module in [(1, baselines), (2, ablations), (3, decoding)]:
        timed(module.__name__.split(".")[-1], lambda: (module.run(config), comparison_tables.export_table(number)))
    timed("knowledge_analysis", lambda: knowledge_analysis.run(config))
    timed("inference_diagnostics", lambda: (diagnostics.run(config), diagnostic_results.plot_inference()))
    timed("reward_alignment", diagnostic_results.reward_alignment)
    timed("case_analysis", case_analysis.run)

    training = pd.concat([pd.read_csv(EXP / f"{name}_training.csv") for name in ["baseline", "ablation"]], ignore_index=True)
    evaluation = pd.concat([pd.read_csv(EXP / f"{name}_evaluations.csv") for name in ["baseline", "ablation", "decoding"]], ignore_index=True)
    write(training, EXP / "training_checks.csv")
    write(evaluation, EXP / "evaluations.csv")
    expected = [f"table{i}.csv" for i in [1, 2, 3, 5]] + ["table4a.csv", "table4b.csv"]
    expected += [f"figure{i}.{suffix}" for i in [3, 4, 5, 7, 8] for suffix in ["png", "pdf"]]
    assert all((RESULTS / name).is_file() and (RESULTS / name).stat().st_size for name in expected)
    assert len(training) == 7 * len(SEEDS) and (training.optimizer_updates > 0).all()
    assert np.isfinite(evaluation[["Sequence", "LCS_F1", "WLCS_F1", "Edge_F1", "Total_Reward"]]).all().all()
    for path in evaluation.Path:
        nodes = path.split(" -> ")
        assert len(nodes) == len(set(nodes))
    evidence = {"status": "PASS", "scope": "synthetic experiment and result workflows", "seeds": SEEDS,
                "training_jobs": len(training), "evaluated_paths": len(evaluation), "timings_seconds": timings,
                "total_seconds": round(time.perf_counter()-started, 3), "outputs": expected}
    (RESULTS / "verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    (RESULTS / "environment.json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    print(f"PASS experiment suite: {evidence['total_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
