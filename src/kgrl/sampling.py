"""Temperature and nucleus sampling extracted from the original inference engine."""
import torch
from kgrl.core.device import get_device

device = get_device()


class PolicySampler:
    def __init__(self, temperature=0.8, top_p=0.85, sample_action_top_k=10):
        self.temperature = temperature
        self.top_p = top_p
        self.sample_action_top_k = sample_action_top_k

    def _sampling_distribution(self, action_scores, temperature=None, top_p=None, sample_action_top_k=None):
        if action_scores is None or action_scores.nelement() == 0:
            empty_indices = torch.empty(0, dtype=torch.long, device=device)
            return action_scores, empty_indices, action_scores

        scores = action_scores.detach().float()
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        base_probs = torch.softmax(scores, dim=0)
        temp = max(float(self.temperature if temperature is None else temperature), 1e-6)
        sample_probs = torch.softmax(scores / temp, dim=0)

        indices = torch.arange(sample_probs.numel(), dtype=torch.long, device=sample_probs.device)

        top_k_value = self.sample_action_top_k if sample_action_top_k is None else sample_action_top_k
        top_k_value = int(top_k_value or 0)
        if 0 < top_k_value < sample_probs.numel():
            values, positions = torch.topk(sample_probs, top_k_value)
            order = torch.argsort(values, descending=True)
            positions = positions[order]
            sample_probs = values[order]
            indices = indices[positions]

        top_p_value = float(self.top_p if top_p is None else top_p)
        if 0.0 < top_p_value < 1.0 and sample_probs.numel() > 1:
            sorted_probs, sorted_positions = torch.sort(sample_probs, descending=True)
            sorted_indices = indices[sorted_positions]
            cumulative = torch.cumsum(sorted_probs, dim=0)
            keep_count = int((cumulative < top_p_value).sum().item()) + 1
            keep_count = max(1, min(keep_count, sorted_probs.numel()))
            sample_probs = sorted_probs[:keep_count]
            indices = sorted_indices[:keep_count]

        total = sample_probs.sum()
        if total <= 0:
            sample_probs = torch.ones_like(sample_probs) / float(sample_probs.numel())
        else:
            sample_probs = sample_probs / total
        return sample_probs, indices, base_probs
