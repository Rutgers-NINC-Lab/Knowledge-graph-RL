"""RotatE scoring helpers for the environment."""
import math

import torch

from kgrl.core.device import get_device


class ScoringMixin:

    def _get_rotate_tensor(self, cache_name, source, key):
        cache = getattr(self, cache_name, None)
        if cache is None:
            cache = {}
            setattr(self, cache_name, cache)

        tensor = cache.get(key)
        dev = get_device()
        if tensor is None or tensor.device != dev:
            tensor = torch.as_tensor(source[key], dtype=torch.float32, device=dev)
            cache[key] = tensor
        return tensor

    def _calculate_rotate_score(self, current_node, relation, next_node):
        """
        Calculate RotatE score for a triple (current_node, relation, next_node).
        Standard RotatE: -|| h * r - t ||, where * is complex multiplication.
        """
        if not getattr(self, "rotate_score_enabled", True):
            return 0.0
        if (
            self.rotate_entity_embeddings is None
            or self.rotate_relation_embeddings is None
        ):
            return 0.0

        if (
            current_node not in self.rotate_entity_embeddings
            or next_node not in self.rotate_entity_embeddings
            or relation not in self.rotate_relation_embeddings
        ):
            return 0.0

        cache = getattr(self, "_rotate_score_cache", None)
        if cache is None:
            cache = {}
            self._rotate_score_cache = cache
        cache_key = (current_node, relation, next_node)
        if cache_key in cache:
            return cache[cache_key]

        try:
            head = self._get_rotate_tensor(
                "_rotate_entity_tensor_cache", self.rotate_entity_embeddings, current_node
            )
            rel = self._get_rotate_tensor(
                "_rotate_relation_tensor_cache", self.rotate_relation_embeddings, relation
            )
            tail = self._get_rotate_tensor(
                "_rotate_entity_tensor_cache", self.rotate_entity_embeddings, next_node
            )
            dim = head.shape[0] // 2
            if dim == 0:
                return 0.0
            h_re, h_im = head[:dim], head[dim:]
            t_re, t_im = tail[:dim], tail[dim:]
            phase = rel[dim:]
            r_re, r_im = torch.cos(phase), torch.sin(phase)
            hr_re = h_re * r_re - h_im * r_im
            hr_im = h_re * r_im + h_im * r_re
            re_diff = hr_re - t_re
            im_diff = hr_im - t_im
            raw = torch.norm(torch.cat([re_diff, im_diff]), p=2)
            raw_norm = raw / math.sqrt(float(dim) * 2.0)
            score = float(torch.tanh(-raw_norm).item())
            cache[cache_key] = score
            return score
        except Exception as e:
            print(f"Error calculating RotatE score: {e}")
            return 0.0
