"""Semiology path reinforcement-learning environment."""

import math
from collections import Counter

from kgrl.env.candidates import CandidateGenerationMixin
from kgrl.env.scoring import ScoringMixin
from kgrl.env.sr_reasoning import SRReasoningMixin


class EnhancedSemiologyPathEnv(CandidateGenerationMixin, SRReasoningMixin, ScoringMixin):
    """
    Environment for generating semiology diffusion trajectories with graph and
    SR-reachability constraints.
    """

    def __init__(self, graph, entity_to_embedding, relation_to_idx, valid_semiologies,
                 temporal_pairs, knowledge_base, rotate_reward_scale=0.2, penalty_scale=2.0,
                 rotate_entity_embeddings=None, rotate_relation_embeddings=None, temporal_scale=4.0,
                 direct_weight=0.7, second_order_weight=0.2, third_order_weight=0.1,
                 relations_trained=False, penalty_decay_rate=0.7,
                 max_steps_per_episode=4,
                 candidate_mode="temporal_first_reachable",
                 temporal_filter_enabled=True,
                 reachability_filter_enabled=True,
                 reachability_max_hops=3,
                 order_bonus_enabled=True,
                 order_bonus_hop1=0.05,
                 order_bonus_hop2=0.02,
                 order_bonus_hop3=0.01,
                 max_candidate_actions=5,
                 temporal_pair_scope="adjacent",
                 rotate_score_enabled=True,
                 anti_hacking_penalty_enabled=False,
                 pair_penalty_scale=0.5,
                 usage_ema_rho=0.995,
                 repeat_penalty_scale=1.0,
                 training_sequences=None,
                 sequence_reward_enabled=False,
                 sequence_reward_scale=2.0,
                 sr_belief_enabled=False,
                 sr_belief_top_m=5,
                 sr_belief_tau=1.0,
                 sr_same_score=0.2,
                 sr_unreachable_score=0.01,
                 sr_belief_min_prob=1e-6):
        self.graph = graph
        self.entity_to_embedding = entity_to_embedding
        self.relation_to_idx = relation_to_idx
        self.idx_to_relation = {v: k for k, v in relation_to_idx.items()}
        self.valid_semiologies = valid_semiologies
        self.temporal_pairs = temporal_pairs
        self.temporal_weights = temporal_pairs
        self.temporal_scale = temporal_scale
        self.temporal_pair_scope = self._normalize_temporal_pair_scope(temporal_pair_scope)
        self.anti_hacking_penalty_enabled = bool(anti_hacking_penalty_enabled)
        self.pair_penalty_scale = float(pair_penalty_scale)
        self.usage_ema_rho = float(usage_ema_rho)
        self.temporal_usage_epsilon = 1e-6
        self.repeat_penalty_scale = float(repeat_penalty_scale)
        self.pair_usage_counts = {}
        self.temporal_usage_ema = {}
        self.training_sequences = tuple(tuple(seq) for seq in (training_sequences or []) if len(seq) >= 2)
        self.sequence_reward_enabled = bool(sequence_reward_enabled)
        self.sequence_reward_scale = float(sequence_reward_scale)
        self.knowledge_base = knowledge_base
        self.sr_belief_enabled = bool(sr_belief_enabled)
        self.sr_belief_top_m = int(sr_belief_top_m)
        self.sr_belief_tau = float(sr_belief_tau)
        self.sr_same_score = float(sr_same_score)
        self.sr_unreachable_score = float(sr_unreachable_score)
        self.sr_belief_min_prob = float(sr_belief_min_prob)
        self.sr_belief_cache = {}
        self.sr_hop_score_cache = {}

        self._configured_rotate_reward_scale = rotate_reward_scale
        self.relations_trained = relations_trained
        if rotate_reward_scale > 0 and not relations_trained:
            print("RotatE disabled: relations_trained=False so rotate_reward_scale forced to 0.")
        if rotate_reward_scale == 0 and relations_trained:
            print("RotatE weight set to 0 by config (RotatE ablation).")
        self.rotate_reward_scale = rotate_reward_scale if relations_trained else 0.0
        self.rotate_score_enabled = bool(rotate_score_enabled)
        self.penalty_scale = penalty_scale
        self.penalty_decay_rate = penalty_decay_rate

        self.max_steps_per_episode = int(max_steps_per_episode)
        self.candidate_mode = str(candidate_mode)
        # Kept for backward-compatible configs/checkpoints. Candidate generation
        # no longer uses temporal filtering.
        self.temporal_filter_enabled = bool(temporal_filter_enabled)
        self.reachability_filter_enabled = bool(reachability_filter_enabled)
        self.reachability_max_hops = int(reachability_max_hops)
        self.order_bonus_enabled = bool(order_bonus_enabled)
        self.order_bonus_hop1 = float(order_bonus_hop1)
        self.order_bonus_hop2 = float(order_bonus_hop2)
        self.order_bonus_hop3 = float(order_bonus_hop3)
        self.max_candidate_actions = int(max_candidate_actions)

        self.direct_weight = direct_weight
        self.second_order_weight = second_order_weight
        self.third_order_weight = third_order_weight

        self.rotate_entity_embeddings = rotate_entity_embeddings
        self.rotate_relation_embeddings = rotate_relation_embeddings
        self._rotate_score_cache = {}
        self._rotate_entity_tensor_cache = {}
        self._rotate_relation_tensor_cache = {}

        self.start_node_str = None
        self.current_node_str = None
        self.current_sr = None
        self.steps_taken = 0
        self.path_history = []
        self.sr_history = []
        self.current_sr_belief = {}
        self.sr_belief_history = []
        self._current_action_to_predicted_sr = {}
        self._current_action_to_order = {}
        self._current_action_to_sr_belief = {}
        self._current_action_to_expected_hop_score = {}
        self.total_steps = 0

    @staticmethod
    def _normalize_temporal_pair_scope(scope):
        scope = str(scope or "adjacent").strip().lower()
        if scope != "adjacent":
            print(f"[WARN] temporal_pair_scope={scope!r} is no longer supported; using adjacent")
        return "adjacent"

    def _iter_path_temporal_pair_candidates(self):
        return list(zip(self.path_history, self.path_history[1:]))

    def _get_path_temporal_pairs(self):
        """Return path pairs that appear in the temporal-pair table."""
        return [
            (src, dst)
            for src, dst in self._iter_path_temporal_pair_candidates()
            if (src, dst) in self.temporal_weights
        ]

    def _count_path_temporal_pairs(self):
        """Count path pairs that appear in the temporal-pair table."""
        return len(self._get_path_temporal_pairs())

    def _count_repeated_nodes(self):
        """Count repeated node occurrences inside the current path."""
        counts = Counter(self.path_history)
        return sum(max(0, count - 1) for count in counts.values())

    def _calculate_anti_hacking_penalty(self, temporal_path_pairs=None):
        if not self.anti_hacking_penalty_enabled:
            return {
                "pair_popularity_raw": 0.0,
                "pair_popularity_penalty": 0.0,
                "pair_overuse_raw": 0.0,
                "pair_overuse_penalty": 0.0,
                "temporal_usage_total": 0.0,
                "repeat_node_count": 0,
                "repeat_node_penalty": 0.0,
                "anti_hacking_penalty": 0.0,
            }

        if temporal_path_pairs is None:
            temporal_path_pairs = self._get_path_temporal_pairs()

        temporal_pair_count = len(self.temporal_weights)
        eps = float(self.temporal_usage_epsilon)
        temporal_usage_total = sum(
            float(self.temporal_usage_ema.get(pair, 0.0))
            for pair in self.temporal_weights
        )
        pair_overuse_raw = 0.0
        if temporal_pair_count > 0:
            target_frequency = 1.0 / float(temporal_pair_count)
            denom = temporal_usage_total + float(temporal_pair_count) * eps
            for pair in temporal_path_pairs:
                usage = float(self.temporal_usage_ema.get(pair, 0.0))
                observed_frequency = (usage + eps) / denom
                overuse_ratio = observed_frequency / target_frequency
                pair_overuse_raw += max(0.0, math.log(max(overuse_ratio, eps)))
        pair_popularity_raw = pair_overuse_raw
        pair_popularity_penalty = float(self.pair_penalty_scale) * pair_overuse_raw

        repeat_node_count = self._count_repeated_nodes()
        repeat_node_penalty = float(self.repeat_penalty_scale) * float(repeat_node_count)
        anti_hacking_penalty = pair_popularity_penalty + repeat_node_penalty

        return {
            "pair_popularity_raw": float(pair_popularity_raw),
            "pair_popularity_penalty": float(pair_popularity_penalty),
            "pair_overuse_raw": float(pair_overuse_raw),
            "pair_overuse_penalty": float(pair_popularity_penalty),
            "temporal_usage_total": float(temporal_usage_total),
            "repeat_node_count": int(repeat_node_count),
            "repeat_node_penalty": float(repeat_node_penalty),
            "anti_hacking_penalty": float(anti_hacking_penalty),
        }

    def _update_pair_usage_counts(self, temporal_path_pairs=None):
        if not self.anti_hacking_penalty_enabled:
            return
        if temporal_path_pairs is None:
            temporal_path_pairs = self._get_path_temporal_pairs()
        batch_counts = Counter(temporal_path_pairs)
        rho = min(max(float(self.usage_ema_rho), 0.0), 0.999999)
        for pair in self.temporal_weights:
            updated = (
                rho * float(self.temporal_usage_ema.get(pair, 0.0))
                + (1.0 - rho) * float(batch_counts.get(pair, 0))
            )
            if updated > 0.0:
                self.temporal_usage_ema[pair] = updated
            elif pair in self.temporal_usage_ema:
                del self.temporal_usage_ema[pair]
        for pair, count in batch_counts.items():
            self.pair_usage_counts[pair] = int(self.pair_usage_counts.get(pair, 0)) + int(count)

    def _calculate_terminal_path_reward_components(self):
        temporal_path_pairs = self._get_path_temporal_pairs()
        temporal_pair_count = len(temporal_path_pairs)
        path_temporal_reward = float(temporal_pair_count) * float(self.temporal_scale)
        anti_hacking = self._calculate_anti_hacking_penalty(temporal_path_pairs)
        sequence_reward = self._calculate_sequence_reward()
        return {
            "temporal_path_pairs": temporal_path_pairs,
            "path_temporal_pair_count": int(temporal_pair_count),
            "path_temporal_reward": float(path_temporal_reward),
            **sequence_reward,
            **anti_hacking,
        }

    @staticmethod
    def _calculate_wlcs_raw(path, reference):
        """Weighted LCS where each continuous matched block of length k scores k^2."""
        n = len(path)
        m = len(reference)
        if n == 0 or m == 0:
            return 0.0

        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        streak = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                if path[i - 1] == reference[j - 1]:
                    streak[i][j] = streak[i - 1][j - 1] + 1
                    for k in range(1, streak[i][j] + 1):
                        dp[i][j] = max(dp[i][j], dp[i - k][j - k] + float(k * k))
        return dp[n][m]

    @staticmethod
    def _calculate_lcs_length(path, reference):
        """Unweighted LCS length used as the minimum evidence gate for sequence reward."""
        n = len(path)
        m = len(reference)
        if n == 0 or m == 0:
            return 0

        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if path[i - 1] == reference[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return int(dp[n][m])

    def _calculate_sequence_reward(self):
        if not self.sequence_reward_enabled or not self.training_sequences:
            return {
                "best_sequence_wlcs_score": 0.0,
                "best_sequence_match_count": 0,
                "sequence_reward": 0.0,
                "best_sequence_index": -1,
                "best_sequence_match": "",
            }

        best_score = 0.0
        best_match_count = 0
        best_index = -1
        best_sequence = ()
        for idx, reference in enumerate(self.training_sequences):
            denom_len = min(len(self.path_history), len(reference))
            if denom_len <= 0:
                continue
            raw_score = self._calculate_wlcs_raw(self.path_history, reference)
            score = raw_score / float(denom_len * denom_len)
            match_count = self._calculate_lcs_length(self.path_history, reference)
            if score > best_score or (score == best_score and match_count > best_match_count):
                best_score = score
                best_match_count = match_count
                best_index = idx
                best_sequence = reference

        # if best_match_count <= 1:
        #    sequence_reward = 0.0
        #else:
        sequence_reward = float(self.sequence_reward_scale) * float(best_score)
        return {
            "best_sequence_wlcs_score": float(best_score),
            "best_sequence_match_count": int(best_match_count),
            "sequence_reward": float(sequence_reward),
            "best_sequence_index": int(best_index),
            "best_sequence_match": " -> ".join(best_sequence) if best_sequence else "",
        }

    def _calculate_path_temporal_reward(self):
        terminal_components = self._calculate_terminal_path_reward_components()
        return (
            terminal_components["path_temporal_pair_count"],
            terminal_components["path_temporal_reward"],
        )

    def reset(self, start_node_str):
        """Reset the environment to a new starting semiology."""
        if start_node_str not in self.valid_semiologies:
            raise ValueError(f"Start node '{start_node_str}' is not in the set of valid semiologies.")

        self.start_node_str = start_node_str
        self.current_node_str = start_node_str
        self.steps_taken = 0
        self.path_history = [start_node_str]

        if self.sr_belief_enabled:
            self.current_sr_belief = self._get_sr_belief(start_node_str)
            self.current_sr = self._belief_top1(self.current_sr_belief)
        else:
            self.current_sr, _order, _score = self._select_sr_for_semiology(start_node_str, previous_sr=None)
            self.current_sr_belief = {self.current_sr: 1.0} if self.current_sr else {}

        self.sr_history = [self.current_sr] if self.current_sr else []
        self.sr_belief_history = [dict(self.current_sr_belief)] if self.current_sr_belief else []
        return self.current_node_str

    def step(self, action_tuple, relation_embedding_tensor):
        """
        Execute an action and return the new state, reward, done flag, and info.
        """
        if len(action_tuple) == 5:
            _, rel_str, next_node_str, _is_connected, _rank_score = action_tuple
        else:
            _, rel_str, next_node_str, _is_connected, _rank_score, _rotate_score = action_tuple
        self.steps_taken += 1
        self.total_steps += 1

        # Step-level reward deliberately excludes temporal signal. Temporal
        # coherence is rewarded once, at path termination, by counting matched
        # temporal pairs according to temporal_pair_scope.
        rotate_score = self._calculate_rotate_score(self.current_node_str, rel_str, next_node_str)
        rotate_reward = float(rotate_score) * self.rotate_reward_scale
        step_penalty = self.penalty_scale * (1.0 - (self.penalty_decay_rate ** float(self.steps_taken)))
        reward = rotate_reward - step_penalty

        previous_sr = self.current_sr
        previous_sr_belief = dict(self.current_sr_belief or {})
        self.current_node_str = next_node_str
        self.path_history.append(next_node_str)

        predicted = None
        chosen_rel = None
        chosen_next = None
        try:
            if len(action_tuple) >= 3:
                _, chosen_rel, chosen_next = action_tuple[:3]
                if not self.sr_belief_enabled:
                    predicted = self._current_action_to_predicted_sr.get((chosen_rel, chosen_next))
        except Exception:
            pass

        if self.sr_belief_enabled:
            next_belief = None
            if chosen_rel is not None and chosen_next is not None:
                next_belief = self._current_action_to_sr_belief.get((chosen_rel, chosen_next))
            if next_belief is None:
                next_belief = self._get_sr_belief(next_node_str)
            self.current_sr_belief = self._update_sr_belief(
                previous_sr_belief,
                next_belief,
            )
            self.current_sr = self._belief_top1(self.current_sr_belief)
            predicted = self.current_sr
        elif predicted is None:
            predicted, _order, _score = self._select_sr_for_semiology(next_node_str, previous_sr=previous_sr)
            self.current_sr = predicted
            self.current_sr_belief = {self.current_sr: 1.0} if self.current_sr else {}
        else:
            self.current_sr = predicted
            self.current_sr_belief = {self.current_sr: 1.0} if self.current_sr else {}

        self.sr_history.append(self.current_sr)
        if self.current_sr_belief:
            self.sr_belief_history.append(dict(self.current_sr_belief))

        done = self.steps_taken >= self.max_steps_per_episode
        no_next_candidates = False
        next_candidates = None
        if not done:
            next_candidates = self.get_candidate_actions()
            no_next_candidates = not bool(next_candidates)
            if no_next_candidates:
                done = True

        path_temporal_pair_count = 0
        path_temporal_reward = 0.0
        pair_overuse_raw = 0.0
        pair_popularity_penalty = 0.0
        temporal_usage_total = 0.0
        repeat_node_count = 0
        repeat_node_penalty = 0.0
        anti_hacking_penalty = 0.0
        best_sequence_wlcs_score = 0.0
        best_sequence_match_count = 0
        sequence_reward = 0.0
        best_sequence_index = -1
        best_sequence_match = ""
        if done:
            terminal_components = self._calculate_terminal_path_reward_components()
            path_temporal_pair_count = terminal_components["path_temporal_pair_count"]
            path_temporal_reward = terminal_components["path_temporal_reward"]
            pair_overuse_raw = terminal_components["pair_overuse_raw"]
            pair_popularity_penalty = terminal_components["pair_popularity_penalty"]
            temporal_usage_total = terminal_components["temporal_usage_total"]
            repeat_node_count = terminal_components["repeat_node_count"]
            repeat_node_penalty = terminal_components["repeat_node_penalty"]
            anti_hacking_penalty = terminal_components["anti_hacking_penalty"]
            best_sequence_wlcs_score = terminal_components["best_sequence_wlcs_score"]
            best_sequence_match_count = terminal_components["best_sequence_match_count"]
            sequence_reward = terminal_components["sequence_reward"]
            best_sequence_index = terminal_components["best_sequence_index"]
            best_sequence_match = terminal_components["best_sequence_match"]
            reward += path_temporal_reward + sequence_reward - anti_hacking_penalty
            self._update_pair_usage_counts(terminal_components["temporal_path_pairs"])

        info = {
            'rotate_score': float(rotate_score),
            'rotate_reward': float(rotate_reward),
            'step_penalty': float(step_penalty),
            'path_temporal_pair_count': int(path_temporal_pair_count),
            'path_temporal_reward': float(path_temporal_reward),
            'pair_overuse_raw': float(pair_overuse_raw),
            'pair_overuse_penalty': float(pair_popularity_penalty),
            'pair_popularity_penalty': float(pair_popularity_penalty),
            'temporal_usage_total': float(temporal_usage_total),
            'repeat_node_count': int(repeat_node_count),
            'repeat_node_penalty': float(repeat_node_penalty),
            'anti_hacking_penalty': float(anti_hacking_penalty),
            'best_sequence_wlcs_score': float(best_sequence_wlcs_score),
            'best_sequence_match_count': int(best_sequence_match_count),
            'sequence_reward': float(sequence_reward),
            'best_sequence_index': int(best_sequence_index),
            'best_sequence_match': best_sequence_match,
            'no_next_candidates': bool(no_next_candidates),
            'next_candidate_actions': next_candidates,
            'sr_belief_enabled': bool(self.sr_belief_enabled),
            'current_sr_belief': dict(self.current_sr_belief),
            'final_reward': float(reward),
        }
        return next_node_str, reward, done, info
