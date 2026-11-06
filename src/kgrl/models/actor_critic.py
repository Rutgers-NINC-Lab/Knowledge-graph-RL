"""Actor-critic network architectures."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kgrl.core.device import get_device


class PathfindingActor(nn.Module):
    def __init__(self, num_relations, entity_dim, relation_dim, gru_hidden_dim, mlp_hidden_dim):
        super().__init__()
        self.relation_embedding = nn.Embedding(num_relations, relation_dim)
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_input_dim = entity_dim + entity_dim
        self.gru = nn.GRU(self.gru_input_dim, gru_hidden_dim, batch_first=True)
        
        self.action_scorer_input_dim = gru_hidden_dim + relation_dim + entity_dim + 1
        # 漏斗形架构解决信息瓶颈：3457 -> 1024 -> 512 -> 128 -> 1
        self.action_scorer = nn.Sequential(
            nn.Linear(self.action_scorer_input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.gru_hidden_dim, device=get_device())

    def score_action_batches(
        self,
        start_entity_emb,
        current_entity_emb,
        hidden_state,
        candidate_action_batches,
        entity_embedding_map,
        current_node_names=None,
    ):
        """Score padded batches of dynamic candidate actions.

        candidate_action_batches is a list with one candidate-action list per state.
        The returned score/prob tensors have shape [batch, max_actions].
        """
        dev = get_device()
        if start_entity_emb.ndim == 1:
            start_entity_emb = start_entity_emb.unsqueeze(0)
        if current_entity_emb.ndim == 1:
            current_entity_emb = current_entity_emb.unsqueeze(0)

        batch_size = start_entity_emb.shape[0]
        if batch_size == 0:
            empty = torch.empty((0, 0), device=dev)
            return empty, empty, hidden_state, torch.empty((0, 0, self.relation_embedding.embedding_dim), device=dev), torch.empty((0, 0), dtype=torch.bool, device=dev)

        gru_input = torch.cat([start_entity_emb, current_entity_emb], dim=-1).unsqueeze(1)
        gru_output, next_hidden_state = self.gru(gru_input, hidden_state)
        gru_output_squeezed = gru_output.squeeze(1)

        max_actions = max((len(actions) for actions in candidate_action_batches), default=0)
        if max_actions == 0:
            empty = torch.empty((batch_size, 0), device=dev)
            return empty, empty, next_hidden_state, torch.empty((batch_size, 0, self.relation_embedding.embedding_dim), device=dev), torch.empty((batch_size, 0), dtype=torch.bool, device=dev)

        rel_indices = torch.zeros((batch_size, max_actions), dtype=torch.long, device=dev)
        action_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool, device=dev)
        next_embeddings = torch.zeros(
            (batch_size, max_actions, self.gru_input_dim // 2),
            dtype=torch.float32,
            device=dev,
        )
        # Reserved scalar feature kept at zero for checkpoint compatibility.
        # Temporal signal is no longer exposed at the action/step level.
        temporal_features = torch.zeros((batch_size, max_actions, 1), dtype=torch.float32, device=dev)

        if current_node_names is None:
            current_node_names = [None] * batch_size
        elif isinstance(current_node_names, str):
            current_node_names = [current_node_names] * batch_size

        for row_idx, actions in enumerate(candidate_action_batches):
            current_node_name = current_node_names[row_idx] if row_idx < len(current_node_names) else None
            for action_idx, action_tuple in enumerate(actions):
                rel_indices[row_idx, action_idx] = int(action_tuple[0])
                next_node_str = action_tuple[2]
                next_emb = entity_embedding_map[next_node_str]
                if isinstance(next_emb, torch.Tensor):
                    next_embeddings[row_idx, action_idx] = next_emb.to(device=dev, dtype=torch.float32)
                else:
                    next_embeddings[row_idx, action_idx] = torch.as_tensor(
                        np.asarray(next_emb),
                        dtype=torch.float32,
                        device=dev,
                    )
                action_mask[row_idx, action_idx] = True

        relation_embeddings = self.relation_embedding(rel_indices)
        gru_features = gru_output_squeezed.unsqueeze(1).expand(-1, max_actions, -1)
        combined_features = torch.cat([gru_features, relation_embeddings, next_embeddings, temporal_features], dim=-1)
        flat_scores = self.action_scorer(combined_features.reshape(batch_size * max_actions, -1))
        scores = flat_scores.view(batch_size, max_actions)
        scores = scores.masked_fill(~action_mask, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1).masked_fill(~action_mask, 0.0)

        return scores, probs, next_hidden_state, relation_embeddings, action_mask

    def forward(
        self,
        start_entity_emb,
        current_entity_emb,
        hidden_state,
        candidate_actions,
        entity_embedding_map,
        current_node_name=None,
    ):
        if not candidate_actions:
            dev = get_device()
            if start_entity_emb.ndim == 1:
                start_entity_emb = start_entity_emb.unsqueeze(0)
            if current_entity_emb.ndim == 1:
                current_entity_emb = current_entity_emb.unsqueeze(0)
            gru_input = torch.cat([start_entity_emb, current_entity_emb], dim=-1).unsqueeze(1)
            _, next_hidden_state = self.gru(gru_input, hidden_state)
            return torch.tensor([], device=dev), torch.tensor([], device=dev), next_hidden_state, []

        scores, probs, next_hidden_state, relation_embeddings, _ = self.score_action_batches(
            start_entity_emb,
            current_entity_emb,
            hidden_state,
            [candidate_actions],
            entity_embedding_map,
            current_node_names=[current_node_name],
        )
        scores_tensor = scores[0, :len(candidate_actions)]
        probs_tensor = probs[0, :len(candidate_actions)]
        # Kept for the existing env.step API; the environment currently does not use it.
        relation_tensors = [relation_embeddings[0, idx] for idx in range(len(candidate_actions))]
        
        return scores_tensor, probs_tensor, next_hidden_state, relation_tensors

class PathfindingCritic(nn.Module):
    def __init__(self, entity_dim, gru_hidden_dim, mlp_hidden_dim):
        super().__init__()
        self.gru_hidden_dim = gru_hidden_dim  # 保存 gru_hidden_dim 供 init_hidden 使用
        self.gru_input_dim = entity_dim + entity_dim
        self.gru = nn.GRU(self.gru_input_dim, gru_hidden_dim, batch_first=True)
        
        # 优化 Critic 的 MLP 架构：256 -> 256 -> 128 -> 1
        self.value_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.gru_hidden_dim, device=get_device())

    def forward(self, start_entity_emb, current_entity_emb, hidden_state):
        if start_entity_emb.ndim == 1: start_entity_emb = start_entity_emb.unsqueeze(0)
        if current_entity_emb.ndim == 1: current_entity_emb = current_entity_emb.unsqueeze(0)
        
        gru_input = torch.cat([start_entity_emb, current_entity_emb], dim=-1).unsqueeze(1)
        gru_output, next_hidden_state = self.gru(gru_input, hidden_state)
        value = self.value_head(gru_output.squeeze(1))
        
        return value.squeeze(-1), next_hidden_state
