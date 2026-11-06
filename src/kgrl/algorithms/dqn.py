"""DQN replay buffer and update logic for dynamic candidate actions."""
import random
from collections import deque

import torch
import torch.nn.functional as F
import torch.nn as nn

from kgrl.core.device import get_device


class DQNReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.storage = deque(maxlen=self.capacity)

    def store(self, state, action, reward, next_state, done):
        self.storage.append({
            "state": state,
            "action": int(action.detach().view(-1)[0].item()) if isinstance(action, torch.Tensor) else int(action),
            "reward": float(reward),
            "next_state": next_state,
            "done": bool(done),
        })

    def sample(self, batch_size):
        sample_size = min(int(batch_size), len(self.storage))
        return random.sample(list(self.storage), sample_size)

    def clear(self):
        self.storage.clear()

    def __len__(self):
        return len(self.storage)


def linear_epsilon(step, epsilon_start, epsilon_end, epsilon_decay_steps):
    if epsilon_decay_steps <= 0:
        return float(epsilon_end)
    progress = min(1.0, float(step) / float(epsilon_decay_steps))
    return float(epsilon_start) + progress * (float(epsilon_end) - float(epsilon_start))


def _squeeze_embedding_tensor(tensor):
    if tensor.ndim > 1:
        return tensor.squeeze(0)
    return tensor


def _normalize_hidden_state(hidden_state):
    if hidden_state is None:
        return None
    if hidden_state.ndim == 2:
        hidden_state = hidden_state.unsqueeze(1)
    return hidden_state


def _state_value(state, key, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return default


def _collect_state_batch(states):
    valid_positions = []
    start_embs = []
    current_embs = []
    hidden_states = []
    candidate_batches = []
    current_node_names = []

    for pos, state in enumerate(states):
        if not state:
            continue
        candidates = _state_value(state, "candidate_actions", [])
        if not candidates:
            continue
        start_emb = _state_value(state, "start_emb")
        current_emb = _state_value(state, "current_emb")
        hidden_state = _normalize_hidden_state(_state_value(state, "hidden_actor"))
        if start_emb is None or current_emb is None or hidden_state is None:
            continue

        valid_positions.append(pos)
        start_embs.append(_squeeze_embedding_tensor(start_emb).to(get_device()))
        current_embs.append(_squeeze_embedding_tensor(current_emb).to(get_device()))
        hidden_states.append(hidden_state.to(get_device()))
        candidate_batches.append(candidates)
        current_node_names.append(_state_value(state, "current_node"))

    if not valid_positions:
        return None

    return {
        "positions": valid_positions,
        "start_batch": torch.stack(start_embs, dim=0),
        "current_batch": torch.stack(current_embs, dim=0),
        "hidden_batch": torch.cat(hidden_states, dim=1),
        "candidate_batches": candidate_batches,
        "current_node_names": current_node_names,
    }


def dqn_update(q_network, target_network, optimizer, replay_buffer, batch_size,
               gamma, entity_embedding_map):
    transitions = replay_buffer.sample(batch_size)
    if not transitions:
        return None

    states = [item["state"] for item in transitions]
    state_batch = _collect_state_batch(states)
    if state_batch is None:
        return None

    valid_transitions = []
    valid_action_indices = []
    for batch_row, transition_idx in enumerate(state_batch["positions"]):
        transition = transitions[transition_idx]
        candidates = _state_value(transition["state"], "candidate_actions", [])
        action_idx = int(transition["action"])
        if action_idx < 0 or action_idx >= len(candidates):
            continue
        valid_transitions.append(transition)
        valid_action_indices.append(action_idx)

    if not valid_transitions:
        return None

    # Re-collect after dropping invalid actions so tensor rows align exactly.
    state_batch = _collect_state_batch([item["state"] for item in valid_transitions])
    if state_batch is None:
        return None

    action_tensor = torch.tensor(valid_action_indices, dtype=torch.long, device=get_device())
    rewards = torch.tensor([item["reward"] for item in valid_transitions], dtype=torch.float32, device=get_device())
    dones = torch.tensor([float(item["done"]) for item in valid_transitions], dtype=torch.float32, device=get_device())

    q_scores, _, _, _, _ = q_network.score_action_batches(
        state_batch["start_batch"],
        state_batch["current_batch"],
        state_batch["hidden_batch"],
        state_batch["candidate_batches"],
        entity_embedding_map,
        current_node_names=state_batch["current_node_names"],
    )
    current_q = q_scores.gather(1, action_tensor.view(-1, 1)).squeeze(1)

    next_values = torch.zeros(len(valid_transitions), dtype=torch.float32, device=get_device())
    next_states = [item["next_state"] for item in valid_transitions]
    next_state_batch = _collect_state_batch(next_states)
    if next_state_batch is not None:
        with torch.no_grad():
            next_q_scores, _, _, _, _ = target_network.score_action_batches(
                next_state_batch["start_batch"],
                next_state_batch["current_batch"],
                next_state_batch["hidden_batch"],
                next_state_batch["candidate_batches"],
                entity_embedding_map,
                current_node_names=next_state_batch["current_node_names"],
            )
            max_next_q = next_q_scores.max(dim=1).values
            for row_idx, transition_pos in enumerate(next_state_batch["positions"]):
                next_values[transition_pos] = max_next_q[row_idx]

    targets = rewards + float(gamma) * (1.0 - dones) * next_values
    loss = F.smooth_l1_loss(current_q, targets.detach())

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_network.parameters(), 1.0)
    optimizer.step()

    return {
        "actor_loss": float(loss.detach().cpu().item()),
        "critic_loss": None,
        "total_loss": float(loss.detach().cpu().item()),
        "entropy": None,
        "mean_q": float(current_q.detach().mean().cpu().item()),
        "mean_target": float(targets.detach().mean().cpu().item()),
    }

