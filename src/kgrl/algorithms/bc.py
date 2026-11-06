"""Original temporal-soft-target BC loss adapted to the current actor API."""
import numpy as np
import torch

def bc_update(actor, optimizer_actor, buffer, epochs, entropy_beta, entity_embedding_map, temporal_weights, label_smoothing=1e-3):
    """Supervised Behavioral Cloning (non-RL) using temporal weights as a soft expert distribution.

    For each stored state (current_node_name, start_emb, current_emb, hidden_state, candidates),
    build target distribution q over candidates:
        q_j ∝ w_temp(current_node, next_node_j)
    and minimize cross-entropy:
        L = -Σ_j q_j log πθ(a_j|s)  - entropy_beta * H(πθ)

    Notes:
    - If none of the candidates has non-zero temporal weight, the sample is skipped.
    - This keeps the rest of the pipeline (candidate generation, rollout, env.step/path_history) unchanged.
    """
    data = buffer.get_data()
    states = data.get('states', [])
    if not states:
        return None

    actor.train()
    losses, entropies = [], []
    used = 0

    for _ in range(int(epochs)):
        for state in states:
            # Expected state tuple:
            # (current_node_name, start_emb_tensor, current_emb_tensor, hidden_state, candidates)
            if not (isinstance(state, tuple) and len(state) == 5):
                continue
            current_node_name, start_emb_tensor, current_emb_tensor, hidden_state, candidates = state
            if not candidates:
                continue

            # Same current actor interface as PPO; temporal targets are used only in the loss.
            logits, probs, _, _ = actor(
                start_emb_tensor, current_emb_tensor, hidden_state, candidates, entity_embedding_map,
                current_node_name=current_node_name
            )
            if probs.nelement() == 0:
                continue

            # Build soft target q over candidates from temporal weights
            q = torch.zeros_like(probs, dtype=torch.float32)
            for j, action_tuple in enumerate(candidates):
                # action tuple formats (len 6 or 8) both include next_node_str at index 2
                try:
                    next_node_str = action_tuple[2]
                except Exception:
                    continue
                q[j] = float(temporal_weights.get((current_node_name, next_node_str), 0.0))

            q_sum = float(q.sum().item())
            if q_sum <= 0.0:
                # no supervision signal for this state
                continue
            q = q / q_sum

            # Optional label smoothing to avoid over-confident collapse
            if label_smoothing and label_smoothing > 0:
                q = (1.0 - label_smoothing) * q + label_smoothing / q.numel()

            logp = torch.log(probs + 1e-8)
            ce = -(q * logp).sum()
            entropy = -(probs * logp).sum()
            loss = ce - float(entropy_beta) * entropy

            optimizer_actor.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer_actor.step()

            losses.append(float(ce.item()))
            entropies.append(float(entropy.item()))
            used += 1

    return {
        "actor_loss": float(np.mean(losses)) if losses else None,
        "critic_loss": None,
        "total_loss": float(np.mean(losses)) if losses else None,
        "entropy": float(np.mean(entropies)) if entropies else None,
        "samples_used": int(used),
    }
