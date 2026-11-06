"""Knowledge-base accessors used by training and inference."""
import csv
from pathlib import Path

import pandas as pd

from kgrl.core.paths import DATA_ROOT


class KnowledgeBase:
    """Enhanced knowledge base for SR-semiology reasoning"""
    
    def __init__(self, data_dir=None):
        # Default to KGRL/data/train_data regardless of current working directory
        self.data_dir = Path(data_dir) if data_dir is not None else (DATA_ROOT / "train_data")
        self.main_func_to_sr = None
        self.func_to_semiology = None
        self.func_conn = None
        self.struc_conn = None
        self.triples_graph = {}
        self._kg_sr_scores_by_semiology = {}
        self._kg_sr_scores_cache = {}
        self._numeric_edges_by_entity = {}
        self.sr_to_semiologies = {}
        self.semiology_to_srs = {}
        self._sr_to_semiologies_sorted = {}
        self._semiology_to_srs_sorted = {}
        self._func_neighbors_by_sr = {}
        self._struc_neighbors_by_sr = {}
        self._both_neighbors_by_sr = {}
        self._triple_related_cache = {}
        self._candidate_srs_cache = {}
        
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load all knowledge base files"""
        print("Loading knowledge base...")
        
        # Load main function to SR mapping (Region column is SR)
        self.main_func_to_sr = pd.read_csv(str(self.data_dir / "f2sr_Sheet1.csv"))
        print(f"Loaded {len(self.main_func_to_sr)} main function to SR mappings")
        
        # Load function to semiology mapping
        self.func_to_semiology = pd.read_csv(str(self.data_dir / "brain_function_symptom_mapping1.csv"))
        print(f"Loaded {len(self.func_to_semiology)} function to semiology mappings")
        
        # Load functional connectivity (Chinese headers)
        self.func_conn = pd.read_csv(str(self.data_dir / "fucn_conn.csv"))
        print(f"Loaded {len(self.func_conn)} functional connections")
        
        # Load structural connectivity
        self.struc_conn = pd.read_csv(str(self.data_dir / "struc_conn_pairwise.csv"))
        print(f"Loaded {len(self.struc_conn)} structural connections")
        
        # Build SR-semiology mappings
        self._build_sr_semiology_mappings()

        # Build connectivity indexes once. Training calls get_connected_srs()
        # inside candidate generation many times; dict lookups avoid repeated
        # pandas filtering in the hot path.
        self._build_connectivity_mappings()
        
        # Load triples graph
        self._load_triples_graph()
        
        print("✅ Knowledge base loaded successfully")
    
    def _build_sr_semiology_mappings(self):
        """Build bidirectional mappings between SRs and semiologies (case/space-insensitive)"""
        # Build SR to semiologies mapping
        for _, row in self.main_func_to_sr.iterrows():
            sr = row['Region']  # Use 'Region' as SR
            function = row['Function']
            
            # Find semiologies for this function (split by comma, strip, lower)
            func_semiologies_raw = self.func_to_semiology[
                self.func_to_semiology['Function'] == function
            ]['Semiology'].tolist()
            func_semiologies = set()
            for semi_str in func_semiologies_raw:
                if isinstance(semi_str, str):
                    for s in semi_str.split(','):
                        s_clean = s.strip().lower()
                        if s_clean:
                            func_semiologies.add(s_clean)
            
            if sr not in self.sr_to_semiologies:
                self.sr_to_semiologies[sr] = set()
            self.sr_to_semiologies[sr].update(func_semiologies)
        
        # Build semiology to SRs mapping (all keys lower/strip)
        for sr, semiologies in self.sr_to_semiologies.items():
            for semiology in semiologies:
                if semiology not in self.semiology_to_srs:
                    self.semiology_to_srs[semiology] = set()
                self.semiology_to_srs[semiology].add(sr)
        
        print(f"Built mappings: {len(self.sr_to_semiologies)} SRs, {len(self.semiology_to_srs)} semiologies")
        self._sr_to_semiologies_sorted = {
            sr: tuple(sorted(semiologies))
            for sr, semiologies in self.sr_to_semiologies.items()
        }
        self._semiology_to_srs_sorted = {
            semiology: tuple(sorted(srs))
            for semiology, srs in self.semiology_to_srs.items()
        }

    @staticmethod
    def _build_neighbor_index(df, left_col, right_col):
        neighbors = {}
        if left_col not in df.columns or right_col not in df.columns:
            return neighbors

        for left, right in df[[left_col, right_col]].itertuples(index=False, name=None):
            if not isinstance(left, str) or not isinstance(right, str):
                continue
            left = left.strip()
            right = right.strip()
            if not left or not right:
                continue
            neighbors.setdefault(left, set()).add(right)
            neighbors.setdefault(right, set()).add(left)

        return {sr: tuple(sorted(vals)) for sr, vals in neighbors.items()}

    def _build_connectivity_mappings(self):
        if len(self.func_conn.columns) >= 3:
            func_left = self.func_conn.columns[1]
            func_right = self.func_conn.columns[2]
            self._func_neighbors_by_sr = self._build_neighbor_index(
                self.func_conn,
                func_left,
                func_right,
            )

        self._struc_neighbors_by_sr = self._build_neighbor_index(
            self.struc_conn,
            "Region 1",
            "Region 2",
        )

        all_srs = set(self._func_neighbors_by_sr) | set(self._struc_neighbors_by_sr)
        both = {}
        for sr in all_srs:
            merged = set(self._func_neighbors_by_sr.get(sr, ()))
            merged.update(self._struc_neighbors_by_sr.get(sr, ()))
            both[sr] = tuple(sorted(merged))
        self._both_neighbors_by_sr = both
    
    def _load_triples_graph(self):
        """Load triples graph for relationship reasoning"""
        triples_file = self.data_dir / "final_triples.csv"
        with open(triples_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    subj, rel, obj = row[0].strip(), row[1].strip(), row[2].strip()
                    if subj not in self.triples_graph:
                        self.triples_graph[subj] = []
                    self.triples_graph[subj].append((rel, obj))

                    try:
                        strength = float(rel)
                    except (TypeError, ValueError):
                        strength = None
                    if strength is None:
                        continue

                    self._add_numeric_edge(subj, obj, strength)

                    canonical_obj_sr = self._canonicalize_sr_name(obj)
                    if canonical_obj_sr is not None:
                        self._add_kg_sr_score(subj, canonical_obj_sr, strength)

                    canonical_subj_sr = self._canonicalize_sr_name(subj)
                    if canonical_subj_sr is not None:
                        self._add_kg_sr_score(obj, canonical_subj_sr, strength)
        
        print(f"Loaded {sum(len(edges) for edges in self.triples_graph.values())} triples")

    def _add_numeric_edge(self, left: str, right: str, strength: float):
        left_key = left.strip().lower()
        right_key = right.strip().lower()
        if not left_key or not right_key:
            return
        left_edges = self._numeric_edges_by_entity.setdefault(left_key, {})
        right_edges = self._numeric_edges_by_entity.setdefault(right_key, {})
        left_edges[right_key] = max(float(strength), float(left_edges.get(right_key, float("-inf"))))
        right_edges[left_key] = max(float(strength), float(right_edges.get(left_key, float("-inf"))))

    def _add_kg_sr_score(self, semiology: str, sr: str, strength: float):
        key = semiology.strip().lower()
        if not key:
            return
        scores = self._kg_sr_scores_by_semiology.setdefault(key, {})
        scores[sr] = max(float(strength), float(scores.get(sr, float("-inf"))))

    def _canonicalize_sr_name(self, sr_name: str):
        """Map KG region labels onto the SR labels used by the KB."""
        if not isinstance(sr_name, str):
            return None
        sr_name = sr_name.strip()
        if sr_name in self.sr_to_semiologies:
            return sr_name
        suffix = " (formerly Supplementary Motor Cortex)"
        if sr_name.endswith(suffix):
            stripped = sr_name[: -len(suffix)].strip()
            if stripped in self.sr_to_semiologies:
                return stripped
        return None
    
    def get_srs_for_semiology(self, semiology: str) -> list:
        """Get SRs associated with a semiology (case/space-insensitive)"""
        key = semiology.strip().lower()
        return list(self._semiology_to_srs_sorted.get(key, ()))

    def get_kg_sr_scores_for_semiology(self, semiology: str) -> dict:
        """Get KG numeric strengths from a semiology to candidate SRs."""
        key = semiology.strip().lower()
        direct_scores = self._kg_sr_scores_by_semiology.get(key, {})
        if direct_scores:
            return dict(direct_scores)
        if key in self._kg_sr_scores_cache:
            return dict(self._kg_sr_scores_cache[key])

        propagated_scores = {}
        for neighbor, edge_score in self._numeric_edges_by_entity.get(key, {}).items():
            neighbor_scores = self._kg_sr_scores_by_semiology.get(neighbor)
            if not neighbor_scores:
                neighbor_scores = {
                    sr: 1.0
                    for sr in self._semiology_to_srs_sorted.get(neighbor, ())
                }
            for sr, sr_score in neighbor_scores.items():
                combined_score = float(edge_score) * float(sr_score)
                propagated_scores[sr] = max(
                    combined_score,
                    float(propagated_scores.get(sr, float("-inf"))),
                )

        self._kg_sr_scores_cache[key] = dict(propagated_scores)
        return dict(propagated_scores)

    def get_strongest_kg_srs_for_semiology(self, semiology: str) -> list:
        """Return the strongest KG SR set for semiologies absent from KB mapping."""
        scores = self.get_kg_sr_scores_for_semiology(semiology)
        if not scores:
            return []
        max_score = max(scores.values())
        return [sr for sr, score in sorted(scores.items()) if score == max_score]

    def get_candidate_srs_for_semiology(self, semiology: str) -> list:
        """
        Return SR candidates for a semiology.

        KB mappings are used when present. If a semiology has no KB mapping,
        derive its SR set from the strongest numeric KG relation(s) to SRs.
        """
        key = semiology.strip().lower()
        if key in self._candidate_srs_cache:
            return list(self._candidate_srs_cache[key])

        kb_srs = self.get_srs_for_semiology(semiology)
        if kb_srs:
            result = tuple(kb_srs)
        else:
            result = tuple(self.get_strongest_kg_srs_for_semiology(semiology))
        self._candidate_srs_cache[key] = result
        return list(result)

    def get_sr_prior_scores_for_semiology(self, semiology: str) -> dict:
        """
        Return layered unnormalized prior scores for SR belief grounding.

        KB mappings are authoritative. KG-derived SRs are used only when the
        semiology has no KB mapping, so high numeric KG edges cannot displace
        curated KB SRs.
        """
        kb_srs = self.get_srs_for_semiology(semiology)
        if kb_srs:
            return {sr: 1.0 for sr in kb_srs}

        kg_scores = self.get_kg_sr_scores_for_semiology(semiology)
        if kg_scores:
            return dict(kg_scores)

        scores = {}
        for rank, sr in enumerate(self.get_candidate_srs_for_semiology(semiology)):
            scores[sr] = 1.0 / float(rank + 1)
        return scores

    def get_kg_sr_score(self, semiology: str, sr: str) -> float:
        """Return numeric KG strength for a semiology-SR pair, or 0 if absent."""
        return float(self.get_kg_sr_scores_for_semiology(semiology).get(sr, 0.0))
    
    def get_semiologies_for_sr(self, sr: str) -> list:
        """Get semiologies associated with an SR"""
        return list(self._sr_to_semiologies_sorted.get(sr, ()))
    
    def get_connected_srs(self, sr: str, connection_type='both') -> list:
        """Get SRs connected to the given SR via functional/structural connectivity"""
        if connection_type == 'func':
            return list(self._func_neighbors_by_sr.get(sr, ()))
        if connection_type == 'struc':
            return list(self._struc_neighbors_by_sr.get(sr, ()))
        if connection_type == 'both':
            return list(self._both_neighbors_by_sr.get(sr, ()))
        return []
    
    def get_triple_related_entities(self, entity: str, relation_type: str = None) -> list:
        """Get entities related to the given entity through triples"""
        cache_key = (entity, relation_type)
        if cache_key in self._triple_related_cache:
            return list(self._triple_related_cache[cache_key])

        if entity not in self.triples_graph:
            return []
        
        related = []
        for rel, obj in self.triples_graph[entity]:
            if relation_type is None or relation_type in rel:
                related.append(obj)
        
        result = tuple(sorted(related))
        self._triple_related_cache[cache_key] = result
        return list(result)
