"""Candidate action generation strategies for the semiology path environment."""

from kgrl.env.sr_belief import expected_transition_score


class CandidateGenerationMixin:
    HOP_SCORE_BY_ORDER = {
        1: 1.0,
        2: 0.65,
        3: 0.35,
    }
    RANK_HOP_WEIGHT = 0.7
    RANK_ROTATE_WEIGHT = 0.3

    @staticmethod
    def _rotate_norm(rotate_score):
        return max(0.0, min(1.0, float(rotate_score) + 1.0))

    def _calculate_rank_score(self, hop_order, rotate_score):
        hop_score = self.HOP_SCORE_BY_ORDER.get(int(hop_order), 0.0)
        return self._calculate_rank_score_from_hop_score(hop_score, rotate_score)

    def _calculate_rank_score_from_hop_score(self, hop_score, rotate_score):
        rotate_norm = self._rotate_norm(rotate_score)
        return (
            self.RANK_HOP_WEIGHT * max(0.0, min(1.0, float(hop_score)))
            + self.RANK_ROTATE_WEIGHT * rotate_norm
        )

    def _is_valid_action_target(self, rel_str, next_node_str):
        return (
            next_node_str in self.valid_semiologies
            and next_node_str not in self.path_history
            and rel_str in self.relation_to_idx
            and next_node_str in self.entity_to_embedding
        )

    def get_candidate_actions(self):
        """
        Generate candidate semiology actions.

        Action tuples keep the historical six-field shape for downstream model
        compatibility, with field 4 as rank_score:
        (relation_id, relation_string, next_node, is_connected, rank_score, rotate_score)
        """
        if self.candidate_mode == "graph_neighbors":
            return self._get_graph_neighbor_actions()

        if self.candidate_mode == "sr_reasoning":
            return self._get_candidate_actions_sr_reasoning()

        # Legacy mode name. Current behavior ranks graph-neighbor candidates by
        # SR-hop score and RotatE score; temporal pairs are not candidate filters.
        return self._get_candidate_actions_temporal_first_reachable()

    def _select_next_sr(self):
        """Select the first predicted target SR without temporal filtering."""
        if not self.current_sr:
            return None, []
        predicted_next_srs = self._predict_next_srs(self.current_sr) or []
        if not predicted_next_srs:
            return None, []
        selected_sr = predicted_next_srs[0]
        return selected_sr, predicted_next_srs

    def _order_for_sr(self, sr):
        direct, second, third = self._get_sr_selection_context(self.current_sr)
        return self._sr_hop_order(sr, direct, second, third)

    def _build_action_tuple(self, rel_str, next_node_str, is_connected, hop_order, hop_score=None):
        rotate_score = self._calculate_rotate_score(self.current_node_str, rel_str, next_node_str)
        if hop_score is None:
            rank_score = self._calculate_rank_score(hop_order, rotate_score)
        else:
            rank_score = self._calculate_rank_score_from_hop_score(hop_score, rotate_score)
        return (
            self.relation_to_idx[rel_str],
            rel_str,
            next_node_str,
            bool(is_connected),
            float(rank_score),
            float(rotate_score),
        )

    def _get_candidate_actions_sr_reasoning(self):
        """SR-first candidate generation: predict SRs -> select first SR -> filter graph actions."""
        if not self.current_sr:
            return []

        selected_sr, _predicted = self._select_next_sr()
        if not selected_sr:
            return []

        selected_sr_semiologies = set(self.knowledge_base.get_semiologies_for_sr(selected_sr))
        selected_order = self._order_for_sr(selected_sr)

        candidate_actions = []
        self._current_action_to_predicted_sr = {}
        self._current_action_to_order = {}
        self._current_action_to_sr_belief = {}
        self._current_action_to_expected_hop_score = {}

        for rel_str, next_node_str in self.graph.get(self.current_node_str, []):
            if next_node_str not in selected_sr_semiologies:
                continue
            if self._is_valid_action_target(rel_str, next_node_str):
                action_tuple = self._build_action_tuple(
                    rel_str,
                    next_node_str,
                    is_connected=True,
                    hop_order=selected_order,
                )
                candidate_actions.append(action_tuple)
                self._current_action_to_predicted_sr[(rel_str, next_node_str)] = selected_sr
                self._current_action_to_order[(rel_str, next_node_str)] = selected_order

        candidate_actions.sort(key=lambda x: -x[4])
        return candidate_actions[:self.max_candidate_actions]

    def _get_candidate_actions_temporal_first_reachable(self):
        """Graph-neighbor candidate generation with SR-hop and RotatE rank score."""
        self._current_action_to_predicted_sr = {}
        self._current_action_to_order = {}
        self._current_action_to_sr_belief = {}
        self._current_action_to_expected_hop_score = {}

        e2_set = {n for _rel, n in self.graph.get(self.current_node_str, [])}
        if not e2_set:
            return []

        best_order_by_e2 = {}
        best_sr_by_e2 = {}
        sr_belief_by_e2 = {}
        expected_hop_by_e2 = {}

        if getattr(self, "sr_belief_enabled", False):
            for e2 in e2_set:
                next_belief = self._get_sr_belief(e2)
                if not next_belief:
                    continue
                best_sr = self._belief_top1(next_belief)
                hop_score = expected_transition_score(
                    getattr(self, "current_sr_belief", {}),
                    next_belief,
                    self.get_sr_transition_score,
                )
                best_sr_by_e2[e2] = best_sr
                best_order_by_e2[e2] = (
                    self._get_sr_hop_order(self.current_sr, best_sr)
                    if self.current_sr and best_sr
                    else 99
                )
                sr_belief_by_e2[e2] = next_belief
                expected_hop_by_e2[e2] = hop_score
        else:
            for e2 in e2_set:
                best_sr, best_ord, _kg_score = self._select_sr_for_semiology(
                    e2,
                    previous_sr=self.current_sr,
                )
                if best_sr is None:
                    continue
                best_sr_by_e2[e2] = best_sr
                best_order_by_e2[e2] = best_ord

        candidate_actions = []
        for rel_str, next_node_str in self.graph.get(self.current_node_str, []):
            if next_node_str not in best_sr_by_e2:
                continue
            if self._is_valid_action_target(rel_str, next_node_str):
                ord_val = best_order_by_e2.get(next_node_str, 99)
                hop_score = expected_hop_by_e2.get(next_node_str)
                action_tuple = self._build_action_tuple(
                    rel_str,
                    next_node_str,
                    is_connected=True,
                    hop_order=ord_val,
                    hop_score=hop_score,
                )
                candidate_actions.append(action_tuple)

                predicted_sr = best_sr_by_e2.get(next_node_str)
                self._current_action_to_predicted_sr[(rel_str, next_node_str)] = predicted_sr
                self._current_action_to_order[(rel_str, next_node_str)] = ord_val
                if next_node_str in sr_belief_by_e2:
                    key = (rel_str, next_node_str)
                    self._current_action_to_sr_belief[key] = dict(sr_belief_by_e2[next_node_str])
                    self._current_action_to_expected_hop_score[key] = float(
                        expected_hop_by_e2.get(next_node_str, 0.0)
                    )

        candidate_actions.sort(key=lambda x: -x[4])
        return candidate_actions[:self.max_candidate_actions]

    def _get_graph_neighbor_actions(self):
        """Graph-neighbor-only candidate generation for ablations."""
        self._current_action_to_predicted_sr = {}
        self._current_action_to_order = {}

        candidate_actions = []
        self._current_action_to_sr_belief = {}
        self._current_action_to_expected_hop_score = {}
        for rel_str, next_node_str in self.graph.get(self.current_node_str, []):
            if self._is_valid_action_target(rel_str, next_node_str):
                action_tuple = self._build_action_tuple(
                    rel_str,
                    next_node_str,
                    is_connected=False,
                    hop_order=99,
                )
                candidate_actions.append(action_tuple)

        candidate_actions.sort(key=lambda x: -x[4])
        return candidate_actions[:self.max_candidate_actions]

    def get_detailed_candidate_analysis(self):
        """Return candidate-analysis details for debugging."""
        if not self.current_sr:
            return {
                'current_sr': None,
                'predicted_next_srs': [],
                'selected_sr': None,
                'selected_sr_semiologies': [],
                'graph_actions': [],
                'candidate_actions': [],
            }

        predicted_next_srs = self._predict_next_srs(self.current_sr)
        if not predicted_next_srs:
            return {
                'current_sr': self.current_sr,
                'predicted_next_srs': [],
                'selected_sr': None,
                'selected_sr_semiologies': [],
                'graph_actions': [],
                'candidate_actions': [],
            }

        selected_sr = predicted_next_srs[0]
        selected_order = self._order_for_sr(selected_sr)
        selected_sr_semiologies = self.knowledge_base.get_semiologies_for_sr(selected_sr)

        graph_actions = []
        for rel_str, next_node_str in self.graph.get(self.current_node_str, []):
            if self._is_valid_action_target(rel_str, next_node_str):
                graph_actions.append((rel_str, next_node_str))

        candidate_actions = []
        selected_sr_semiology_set = set(selected_sr_semiologies)
        for rel_str, next_node_str in graph_actions:
            if next_node_str not in selected_sr_semiology_set:
                continue
            action_tuple = self._build_action_tuple(
                rel_str,
                next_node_str,
                is_connected=True,
                hop_order=selected_order,
            )
            candidate_actions.append(action_tuple)

        candidate_actions.sort(key=lambda x: -x[4])
        candidate_actions = candidate_actions[:self.max_candidate_actions]

        return {
            'current_sr': self.current_sr,
            'predicted_next_srs': predicted_next_srs,
            'selected_sr': selected_sr,
            'selected_sr_semiologies': list(selected_sr_semiologies),
            'graph_actions': graph_actions,
            'candidate_actions': candidate_actions,
        }
