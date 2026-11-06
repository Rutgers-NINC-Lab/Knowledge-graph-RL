"""SR reachability and temporal reasoning helpers."""

from kgrl.env.sr_belief import prune_belief


class SRReasoningMixin:

    def _get_candidate_srs_for_semiology(self, semiology):
        return self.knowledge_base.get_candidate_srs_for_semiology(semiology)

    @staticmethod
    def _belief_top1(belief):
        if not belief:
            return None
        return sorted(belief.items(), key=lambda item: (-item[1], item[0]))[0][0]

    def get_sr_belief(self, semiology, top_k=None, tau=None):
        """Return q(z | semiology), a cached layered SR belief distribution."""
        top_k = int(top_k if top_k is not None else getattr(self, "sr_belief_top_m", 5))
        tau = float(tau if tau is not None else getattr(self, "sr_belief_tau", 1.0))
        min_prob = float(getattr(self, "sr_belief_min_prob", 1e-6))
        cache_key = (str(semiology), top_k, tau, min_prob)

        cache = getattr(self, "sr_belief_cache", None)
        if cache is None:
            self.sr_belief_cache = {}
            cache = self.sr_belief_cache
        if cache_key in cache:
            return dict(cache[cache_key])

        kb_srs = self.knowledge_base.get_srs_for_semiology(semiology)
        if kb_srs:
            prob = 1.0 / float(len(kb_srs))
            belief = {sr: prob for sr in sorted(kb_srs)}
        else:
            kg_scores = self.knowledge_base.get_kg_sr_scores_for_semiology(semiology)
            if kg_scores:
                belief = self._normalize_sr_scores(
                    dict(
                        sorted(
                            ((sr, float(score)) for sr, score in kg_scores.items()),
                            key=lambda item: (-item[1], item[0]),
                        )[: max(1, top_k)]
                    ),
                    min_prob=min_prob,
                )
            else:
                fallback_scores = self.knowledge_base.get_sr_prior_scores_for_semiology(semiology)
                belief = self._normalize_sr_scores(
                    dict(
                        sorted(
                            ((sr, float(score)) for sr, score in fallback_scores.items()),
                            key=lambda item: (-item[1], item[0]),
                        )[: max(1, top_k)]
                    ),
                    min_prob=min_prob,
                )
        cache[cache_key] = dict(belief)
        return dict(belief)

    @staticmethod
    def _normalize_sr_scores(scores, min_prob=1e-6):
        if not scores:
            return {}

        positive_scores = {
            sr: max(float(score), 0.0)
            for sr, score in scores.items()
        }
        total = sum(positive_scores.values())
        if total <= 0.0:
            prob = 1.0 / float(len(positive_scores))
            belief = {sr: prob for sr in sorted(positive_scores)}
        else:
            belief = {
                sr: value / total
                for sr, value in positive_scores.items()
            }
        return prune_belief(
            belief,
            top_m=len(belief),
            min_prob=min_prob,
        )

    def _get_sr_belief(self, semiology):
        return self.get_sr_belief(
            semiology,
            top_k=getattr(self, "sr_belief_top_m", 5),
            tau=getattr(self, "sr_belief_tau", 1.0),
        )

    def get_sr_transition_score(self, sr_from, sr_to):
        """Return cached H(sr_from, sr_to) transition compatibility."""
        if not sr_from or not sr_to:
            return float(getattr(self, "sr_unreachable_score", 0.01))

        cache_key = (str(sr_from), str(sr_to))
        cache = getattr(self, "sr_hop_score_cache", None)
        if cache is None:
            self.sr_hop_score_cache = {}
            cache = self.sr_hop_score_cache
        if cache_key in cache:
            return float(cache[cache_key])

        if sr_from == sr_to:
            score = float(getattr(self, "sr_same_score", 0.2))
        else:
            hop_order = self._get_sr_hop_order(sr_from, sr_to)
            if hop_order == 1:
                score = 1.0
            elif hop_order == 2:
                score = 0.65
            elif hop_order == 3:
                score = 0.35
            else:
                score = float(getattr(self, "sr_unreachable_score", 0.01))

        cache[cache_key] = float(score)
        return float(score)

    def _update_sr_belief(self, prev_belief, obs_belief):
        """
        Transition-aware belief update:
        b_{t+1}(z') ∝ q(z' | s_{t+1}) * sum_z b_t(z) * psi(z, z').
        """
        if not prev_belief:
            return dict(obs_belief or {})
        if not obs_belief:
            return {}

        updated = {}
        for next_sr, obs_prob in obs_belief.items():
            transition_mass = 0.0
            for prev_sr, prev_prob in prev_belief.items():
                transition_mass += (
                    float(prev_prob)
                    * self.get_sr_transition_score(prev_sr, next_sr)
                )
            updated[next_sr] = float(obs_prob) * transition_mass

        total = sum(updated.values())
        if total <= 0.0:
            return dict(obs_belief)

        normalized = {
            sr: float(value) / float(total)
            for sr, value in updated.items()
        }
        return prune_belief(
            normalized,
            top_m=len(normalized),
            min_prob=getattr(self, "sr_belief_min_prob", 1e-6),
        )

    def _get_sr_selection_context(self, previous_sr):
        if not previous_sr or not getattr(self, "reachability_filter_enabled", True):
            return set(), set(), set()
        reachable = self._get_reachable_srs_by_order(previous_sr)
        direct = set(reachable['direct'])
        second = set(reachable['second']) if self.reachability_max_hops >= 2 else set()
        third = set(reachable['third']) if self.reachability_max_hops >= 3 else set()
        direct.add(previous_sr)
        return direct, second, third

    @staticmethod
    def _sr_hop_order(sr, direct, second, third):
        if sr in direct:
            return 1
        if sr in second:
            return 2
        if sr in third:
            return 3
        return 99

    def _get_sr_hop_order(self, prev_sr, next_sr):
        direct, second, third = self._get_sr_selection_context(prev_sr)
        return self._sr_hop_order(next_sr, direct, second, third)

    def _select_sr_for_semiology(self, semiology, previous_sr=None):
        """
        Ground a semiology to one SR using unified KB/KG candidates.

        Candidate SRs come from KB when available; otherwise the strongest
        numeric KG semiology->SR relation(s) are used. A previous SR, when
        available, ranks candidates by direct/second/third-order reachability.
        """
        candidate_srs = self._get_candidate_srs_for_semiology(semiology)
        if not candidate_srs:
            return None, 99, 0.0

        direct, second, third = self._get_sr_selection_context(previous_sr)
        ranked = []
        for sr in candidate_srs:
            order = self._sr_hop_order(sr, direct, second, third)
            kg_score = self.knowledge_base.get_kg_sr_score(semiology, sr)
            ranked.append((order, -float(kg_score), sr))
        ranked.sort()
        order, neg_score, sr = ranked[0]
        return sr, order, -neg_score

    def _predict_next_srs(self, current_sr):
        """Predict next SRs using multi-order SR graph connections only."""
        # For the first step, only direct
        if self.steps_taken == 0:
            return self._predict_next_srs_direct_only(current_sr)
        return self._predict_next_srs_multi_order(current_sr)

    def _predict_next_srs_direct_only(self, current_sr, use_temporal=None):
        """Predict next SRs via direct func/struc connections."""
        direct_func = self.knowledge_base.get_connected_srs(current_sr, 'func')
        direct_struc = self.knowledge_base.get_connected_srs(current_sr, 'struc')
        direct_srs = sorted(set(direct_func + direct_struc))
        return direct_srs or [current_sr]

    def _get_second_order_srs(self, current_sr, use_temporal=None):
        second = set()
        direct = self._predict_next_srs_direct_only(current_sr)
        for d in direct:
            if d == current_sr:
                continue
            second.update(self.knowledge_base.get_connected_srs(d, 'func'))
            second.update(self.knowledge_base.get_connected_srs(d, 'struc'))
        second.discard(current_sr)
        for d in direct:
            second.discard(d)
        return sorted(second)

    def _get_third_order_srs(self, current_sr, use_temporal=None):
        third = set()
        second = self._get_second_order_srs(current_sr)
        for s in second:
            third.update(self.knowledge_base.get_connected_srs(s, 'func'))
            third.update(self.knowledge_base.get_connected_srs(s, 'struc'))
        direct = self._predict_next_srs_direct_only(current_sr)
        third.discard(current_sr)
        for d in direct:
            third.discard(d)
        for s in second:
            third.discard(s)
        return sorted(third)

    def _predict_next_srs_multi_order(self, current_sr, use_temporal=None):
        direct = self._predict_next_srs_direct_only(current_sr)
        second = self._get_second_order_srs(current_sr)
        third = self._get_third_order_srs(current_sr)
        weighted = {}

        for sr in direct:
            weighted[sr] = weighted.get(sr, 0) + self.direct_weight
        for sr in second:
            weighted[sr] = weighted.get(sr, 0) + self.second_order_weight
        for sr in third:
            weighted[sr] = weighted.get(sr, 0) + self.third_order_weight
        return [sr for sr, _w in sorted(weighted.items(), key=lambda kv: (-kv[1], kv[0]))] or [current_sr]

    def _get_reachable_srs_by_order(self, current_sr):
        direct = set(self._predict_next_srs_direct_only(current_sr))
        second = set(self._get_second_order_srs(current_sr))
        third = set(self._get_third_order_srs(current_sr))
        direct.discard(current_sr)
        second.discard(current_sr)
        third.discard(current_sr)
        return {'direct': direct, 'second': second, 'third': third}

    def get_next_sr_candidates(self, next_node_str):
        """获取下一个节点的SR候选"""
        new_srs = self._get_candidate_srs_for_semiology(next_node_str)
        if not new_srs:
            return {
                'all_candidates': [],
                'preferred': [],
                'non_preferred': []
            }
        
        # 如果有当前SR，优先选择连接的SR
        if self.current_sr:
            connected_srs = set(self.knowledge_base.get_connected_srs(self.current_sr, 'both'))
            connected_srs.add(self.current_sr)
            preferred = [sr for sr in new_srs if sr in connected_srs]
            non_preferred = [sr for sr in new_srs if sr not in connected_srs]
        else:
            preferred = []
            non_preferred = new_srs
        
        return {
            'all_candidates': new_srs,
            'preferred': preferred,
            'non_preferred': non_preferred
        }
