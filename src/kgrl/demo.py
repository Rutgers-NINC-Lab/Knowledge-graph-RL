"""PPO training, reward-ranked decoding and sequence evaluation for the small demo."""
import json
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

from kgrl.algorithms.ppo import PPOBuffer, ppo_update
from kgrl.algorithms.dqn import DQNReplayBuffer, dqn_update, linear_epsilon
from kgrl.algorithms.bc import bc_update
from kgrl.data.knowledge_base import KnowledgeBase
from kgrl.env.semiology_env import EnhancedSemiologyPathEnv
from kgrl.models.actor_critic import PathfindingActor, PathfindingCritic
from kgrl.sampling import PolicySampler
from kgrl import metrics
from kgrl.core.device import get_device


class Demo:
    def __init__(self, root, config):
        self.root, self.config = root, config
        self.device = get_device()
        self.sample = json.loads((root / "data/sample.json").read_text(encoding="utf-8"))
        self.nodes = self.sample["nodes"]
        self.relations = {r: i for i, r in enumerate(self.sample["relations"])}
        self.graph = {n: [] for n in self.nodes}
        for head, relation, tail in self.sample["edges"]:
            self.graph[head].append((relation, tail))
        with np.load(root / "data/embeddings.npz") as arrays:
            self.embeddings = {n: arrays["entity"][i].copy() for i, n in enumerate(self.nodes)}
            self.rotate_entities = {n: arrays["rotate_entity"][i].copy() for i, n in enumerate(self.nodes)}
            self.rotate_relations = {r: arrays["rotate_relation"][i].copy() for r, i in self.relations.items()}
        self.tensors = {n: torch.as_tensor(v, device=self.device) for n, v in self.embeddings.items()}
        self.temporal = {(a, b): w for a, b, w in self.sample["temporal_pairs"]}
        self.kb = KnowledgeBase(root / "data/train_data")

    def environment(self, stage=2, max_steps=None):
        valid = sorted({n for pair in self.temporal for n in pair}) if stage == 1 else self.nodes
        return EnhancedSemiologyPathEnv(
            self.graph, self.embeddings, self.relations, valid, self.temporal, self.kb,
            rotate_entity_embeddings=self.rotate_entities, rotate_relation_embeddings=self.rotate_relations,
            # This source flag enables the scorer; these demo vectors are random, not pretrained.
            relations_trained=True, training_sequences=self.sample["training_sequences"],
            max_steps_per_episode=max_steps or self.config["training"]["max_steps_per_episode"],
            **self.config["environment"],
        )

    def actor(self):
        return PathfindingActor(len(self.relations), **self.config["model"]).to(self.device)

    def train(self, out, algorithm="ppo", curriculum=True):
        cfg = self.config["training"]
        actor = self.actor()
        model = self.config["model"]
        critic = PathfindingCritic(model["entity_dim"], model["gru_hidden_dim"], model["mlp_hidden_dim"]).to(self.device)
        initial = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        opt_a = torch.optim.Adam(actor.parameters(), lr=cfg["learning_rate"])
        opt_c = torch.optim.Adam(critic.parameters(), lr=cfg["learning_rate"])
        buffer = DQNReplayBuffer(10000) if algorithm == "dqn" else PPOBuffer(cfg["ppo_batch_size"])
        target = deepcopy(actor).eval() if algorithm == "dqn" else None
        node_ema, pair_ema = {}, {}
        losses, episodes = [], []
        stage = 1 if curriculum else 2
        env = self.environment(stage=stage)
        transitions = 0

        def update(episode):
            if algorithm == "dqn":
                result = dqn_update(actor, target, opt_a, buffer, cfg["ppo_batch_size"], 0.99, self.tensors)
            elif algorithm == "bc":
                result = bc_update(actor, opt_a, buffer, cfg["ppo_epochs"], cfg["entropy_beta"], self.tensors, self.temporal)
            else:
                result = ppo_update(
                actor, critic, opt_a, opt_c, buffer, cfg["ppo_epochs"],
                cfg["ppo_clip_epsilon"], cfg["entropy_beta"], self.tensors,
                anti_hub_loss_enabled=cfg["anti_hub_loss_enabled"],
                anti_hub_node_lambda=cfg["anti_hub_node_lambda"],
                anti_hub_pair_lambda=cfg["anti_hub_pair_lambda"],
                anti_hub_node_adv_ema=node_ema, anti_hub_pair_adv_ema=pair_ema,
                anti_hub_ema_rho=cfg["anti_hub_ema_rho"],
            )
            if result is None:
                raise RuntimeError("PPO produced no optimizer update")
            if result.get("total_loss") is not None:
                losses.append({"update": len(losses) + 1, "episode": episode, **result})
            if algorithm != "dqn":
                buffer.clear()

        switch = int(cfg["num_episodes"] * cfg["curriculum_switch_fraction"])
        for episode in range(cfg["num_episodes"]):
            if curriculum and episode == switch:
                previous = env
                env = self.environment(stage=2)
                env.pair_usage_counts = previous.pair_usage_counts.copy()
                env.temporal_usage_ema = previous.temporal_usage_ema.copy()
                stage = 2
            start = random.choice(env.valid_semiologies)
            current = env.reset(start)
            ha, hc = actor.init_hidden(), critic.init_hidden()
            total = 0.0
            for step in range(cfg["max_steps_per_episode"]):
                actions = env.get_candidate_actions()
                if not actions:
                    break
                state = {"start_emb": self.tensors[start], "current_emb": self.tensors[current],
                         "current_node": current, "hidden_actor": ha.detach().clone(),
                         "hidden_critic": hc.detach().clone(), "candidate_actions": actions}
                with torch.no_grad():
                    scores, probs, next_ha, relations = actor(self.tensors[start], self.tensors[current], ha,
                                                        actions, self.tensors, current_node_name=current)
                    value, next_hc = critic(self.tensors[start], self.tensors[current], hc)
                    dist = Categorical(probs=probs)
                    if algorithm == "dqn":
                        epsilon = linear_epsilon(transitions, 1.0, 0.05, 5000)
                        action = torch.tensor(random.randrange(len(actions)) if random.random() < epsilon else int(scores.argmax()), device=self.device)
                    else:
                        action = dist.sample()
                    logprob = dist.log_prob(action)
                current, reward, done, info = env.step(actions[action.item()], relations[action.item()])
                with torch.no_grad():
                    next_value = torch.tensor(0.0, device=self.device) if done else critic(self.tensors[start], self.tensors[current], next_hc)[0].squeeze()
                if algorithm == "dqn":
                    next_state = None if done else {"start_emb": self.tensors[start], "current_emb": self.tensors[current],
                                "current_node": current, "hidden_actor": next_ha.detach().clone(),
                                "candidate_actions": info["next_candidate_actions"]}
                    buffer.store(state, action, reward, next_state, done)
                else:
                    if algorithm == "bc":
                        state = (state["current_node"], state["start_emb"], state["current_emb"], state["hidden_actor"], actions)
                    buffer.store(state, action, logprob, reward, value, next_value, done)
                ha, hc = next_ha, next_hc
                total += reward
                transitions += 1
                if len(buffer) >= cfg["ppo_batch_size"]:
                    update(episode + 1)
                if algorithm == "dqn" and transitions % 10 == 0:
                    target.load_state_dict(actor.state_dict())
                if done:
                    break
            episodes.append({"episode": episode + 1, "stage": stage, "reward": total, "steps": env.steps_taken})
        # A singleton cannot be standardized by the original sample-SD GAE normalization.
        if algorithm != "dqn" and len(buffer) >= 2:
            update(cfg["num_episodes"])
        parameter_change = sum(float((v - initial[k]).abs().sum()) for k, v in actor.state_dict().items())
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "config": self.config,
                    "env_state": {"pair_usage_counts": env.pair_usage_counts, "temporal_usage_ema": env.temporal_usage_ema}},
                   out / "model.pt")
        pd.DataFrame(episodes).to_csv(out / "training.csv", index=False)
        pd.DataFrame(losses).to_csv(out / "losses.csv", index=False)
        return {"episodes": len(episodes), "transitions": transitions, "optimizer_updates": len(losses),
                "actor_parameter_l1_change": parameter_change,
                "unused_final_transitions": len(buffer) if algorithm != "dqn" else 0}

    def infer(self, out, mode="policy", max_steps=None, num_samples=None, starts=None, write=True):
        checkpoint = torch.load(out / "model.pt", map_location=self.device, weights_only=True)
        assert checkpoint["config"] == self.config, "Checkpoint/config mismatch"
        actor = self.actor()
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()
        cfg = self.config["inference"]
        sampler = PolicySampler(cfg["temperature"], cfg["top_p"], cfg["sample_action_top_k"])
        max_steps = max_steps or cfg["max_steps"]
        num_samples = num_samples or cfg["num_samples"]
        env = self.environment(max_steps=max_steps)
        rows = []
        pool = []
        for start in self.nodes if starts is None else starts:
            candidates = []
            for _ in range(1 if mode == "greedy" else num_samples):
                # Freeze training usage statistics so ranking is independent of rollout order.
                for name, value in checkpoint["env_state"].items():
                    setattr(env, name, value.copy())
                current = env.reset(start)
                hidden = actor.init_hidden()
                total, logprob, terminal = 0.0, 0.0, {}
                for _ in range(max_steps):
                    actions = env.get_candidate_actions()
                    if not actions:
                        break
                    with torch.no_grad():
                        scores, _, hidden, relations = actor(self.tensors[start], self.tensors[current], hidden,
                                                            actions, self.tensors, current_node_name=current)
                        if mode in {"score", "greedy"}:
                            scores = torch.tensor([a[4] for a in actions], device=self.device)
                        elif mode == "random":
                            scores = torch.zeros(len(actions), device=self.device)
                        probs, indices, base = sampler._sampling_distribution(scores)
                        if mode == "random":
                            index = random.randrange(len(actions))
                        elif mode == "greedy":
                            index = int(scores.argmax())
                        else:
                            index = int(indices[torch.multinomial(probs, 1)].item())
                        logprob += float(base[index].clamp_min(1e-12).log())
                    current, reward, done, terminal = env.step(actions[index], relations[index])
                    total += reward
                    if done:
                        break
                candidates.append({"Start_Semiology": start, "Path": " -> ".join(env.path_history),
                                   "SR_Sequence": " -> ".join(env.sr_history), "Total_Reward": total,
                                   "LogProb": logprob, "Temporal_Pairs": terminal.get("path_temporal_pair_count", 0),
                                   "Training_WLCS": terminal.get("best_sequence_wlcs_score", 0.0)})
            pool.extend({"Sample": i + 1, "Rank": i + 1, **c} for i, c in enumerate(candidates))
            candidates.sort(key=lambda c: (-round(c["Total_Reward"], 8), -c["Temporal_Pairs"],
                                           -round(c["Training_WLCS"], 8), -c["LogProb"], c["Path"], c["SR_Sequence"]))
            seen = set()
            for candidate in candidates:
                if candidate["SR_Sequence"] in seen:
                    continue
                seen.add(candidate["SR_Sequence"])
                rows.append({"Rank": len(seen), **candidate})
                if len(seen) == cfg["top_k"]:
                    break
        predictions = pd.DataFrame(rows)
        self.last_candidates = pd.DataFrame(pool)
        if write:
            predictions.to_csv(out / "predictions.csv", index=False)
            self.last_candidates.to_csv(out / "candidate_pool.csv", index=False)
        return predictions

    def evaluate(self, predictions, out, write=True):
        # References are opened only after candidate selection has finished.
        refs, vocab = metrics.load_reference_rows(str(self.root / "data/references.csv"), "Path", True, "info", 3.0, 0)
        rows = []
        for row in predictions.to_dict("records"):
            nodes = metrics.split_path_to_nodes(row["SR_Sequence"], True)
            scored = [(ref, metrics.score_candidate(nodes, ref, 0.0, vocab, False)) for ref in refs]
            ref, score = max(scored, key=lambda x: (x[1]["final_score"], x[1]["wlcs_score"], x[1]["lcs_score"], x[1]["edge_f1"], -x[0].row_idx))
            rows.append({**row, "Best_Reference": ref.repr_path, "Sequence": score["sequence_score"],
                         "LCS_F1": score["lcs_score"], "WLCS_F1": score["wlcs_score"], "Edge_F1": score["edge_f1"]})
        evaluated = pd.DataFrame(rows)
        if not write:
            return evaluated
        evaluated.to_csv(out / "evaluation.csv", index=False)
        summaries = []
        for k in sorted({1, 3, self.config["inference"]["top_k"]}):
            group = evaluated[evaluated.Rank <= k].groupby("Start_Semiology")
            summaries.append({"Top_k": k, "Starts": len(group), **group[["Sequence", "LCS_F1", "WLCS_F1", "Edge_F1"]].mean().mean().to_dict()})
        pd.DataFrame(summaries).to_csv(out / "metrics.csv", index=False)
        return evaluated
