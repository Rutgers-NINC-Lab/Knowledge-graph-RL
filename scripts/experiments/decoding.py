"""Table 3 experiment: PPO beam decoding versus reward-ranked PPO paths."""
import copy
import pandas as pd
from scripts.experiments.common import ROOT, EXP, SEEDS, Demo, evaluate, prepare, require_baselines, set_seed, torch, write


def beam_decode(demo, directory, policy=True):
    """Small beam search under the shared environment; rank by accumulated log probability."""
    checkpoint = torch.load(directory / "model.pt", map_location=demo.device, weights_only=True)
    actor = demo.actor()
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    rows = []
    for start in demo.nodes:
        max_steps = demo.config["inference"]["max_steps"]
        env = demo.environment(max_steps=max_steps)
        for key, value in checkpoint["env_state"].items():
            setattr(env, key, value.copy())
        env.reset(start)
        beam = [(env, actor.init_hidden(), 0.0, 0.0, False)]
        for _ in range(max_steps):
            expanded = []
            for state, hidden, reward, logprob, done in beam:
                actions = state.get_candidate_actions()
                if done or not actions:
                    expanded.append((state, hidden, reward, logprob, True))
                    continue
                with torch.no_grad():
                    logits, _, next_hidden, _ = actor(demo.tensors[start], demo.tensors[state.current_node_str],
                                                    hidden, actions, demo.tensors)
                    if not policy:
                        logits = torch.tensor([a[4] for a in actions], device=demo.device)
                    log_probs = logits.log_softmax(0)
                for i, action in enumerate(actions):
                    child = copy.deepcopy(state)
                    _, delta, child_done, _ = child.step(action, None)
                    expanded.append((child, next_hidden, reward + delta, logprob + float(log_probs[i]), child_done))
            beam = sorted(expanded, key=lambda x: (-x[3], " -> ".join(x[0].path_history)))[:demo.config["inference"]["top_k"]]
        seen = set()
        for state, _, reward, logprob, _ in beam:
            sr = " -> ".join(state.sr_history)
            if sr not in seen:
                seen.add(sr)
                rows.append({"Rank": len(seen), "Start_Semiology": start, "Path": " -> ".join(state.path_history),
                             "SR_Sequence": sr, "Total_Reward": reward, "LogProb": logprob})
    return pd.DataFrame(rows)


def run(config):
    require_baselines()
    seed = SEEDS[0]  # The compact workflow evaluates this deterministic decoder once.
    cfg = copy.deepcopy(config)
    cfg["seed"] = seed
    set_seed(seed + 1)
    demo = Demo(ROOT, cfg)
    directory = EXP / "models" / f"ppo_seed{seed}"
    scored = evaluate(demo, beam_decode(demo, directory), directory, "PPO beam search", seed)
    write(scored, EXP / "decoding_evaluations.csv")


if __name__ == "__main__":
    run(prepare())
    from scripts.results.comparison_tables import export_table
    export_table(3)
