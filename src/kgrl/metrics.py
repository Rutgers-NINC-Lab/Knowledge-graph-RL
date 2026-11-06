"""Original KGRL sequence metrics, without the batch comparison CLI."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple
import pandas as pd

@dataclass(frozen=True)
class ReferenceEntry:
    row_idx: int
    layers: List[Set[str]]          # reference 的“层”（每层是一个候选集合）
    weights: List[float]            # 每层权重
    total_weight: float             # 权重和
    repr_path: str                  # 规范化后的可读 path（A -> B/C -> D）


def normalize_node(node: str, drop_epilepsy: bool = True) -> str:
    """清洗节点字符串（可选去掉 ' epilepsy' 后缀，并压缩空格）。"""
    if node is None:
        return ""
    clean = str(node).strip()
    if drop_epilepsy:
        clean = clean.replace(" epilepsy", "").strip()
    clean = " ".join(clean.split())
    return clean


def split_path_to_nodes(path_str: str, drop_epilepsy: bool) -> List[str]:
    """pred: 'A -> B -> C' -> ['A','B','C']（不支持层内多选）"""
    if path_str is None or (isinstance(path_str, float) and math.isnan(path_str)):
        return []
    text = str(path_str).strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split("->")]
    nodes: List[str] = []
    for p in parts:
        if not p:
            continue
        n = normalize_node(p, drop_epilepsy)
        if n:
            nodes.append(n)
    return nodes


def parse_reference_layers(path_str: str, drop_epilepsy: bool) -> List[Set[str]]:
    """
    reference: 'A -> B/C -> D' -> [{'A'},{'B','C'},{'D'}]
    """
    if path_str is None or (isinstance(path_str, float) and math.isnan(path_str)):
        return []
    text = str(path_str).strip()
    if not text:
        return []
    steps = [step.strip() for step in text.split("->")]
    layers: List[Set[str]] = []
    for step in steps:
        if not step:
            continue
        parts = []
        for p in step.split("/"):
            n = normalize_node(p, drop_epilepsy)
            if n:
                parts.append(n)
        if parts:
            layers.append(set(parts))
    return layers


def layers_to_repr(layers: Sequence[Set[str]]) -> str:
    """把 layers 还原成可读字符串：[{A},{B,C},{D}] -> 'A -> B/C -> D'"""
    parts: List[str] = []
    for layer in layers:
        if not layer:
            continue
        if len(layer) == 1:
            parts.append(next(iter(layer)))
        else:
            parts.append("/".join(sorted(layer)))
    return " -> ".join(parts)


def layer_weights(layers: Sequence[Set[str]], vocab_size: int, mode: str, alpha: float) -> Tuple[List[float], float]:
    """
    mode:
      - info:  w = -log(|S|/|V|)
      - power: w = 1/|S|^alpha
    """
    weights: List[float] = []
    denom = max(vocab_size, 1)
    for layer in layers:
        size = max(len(layer), 1)
        if mode == "power":
            w = 1.0 / (size ** alpha)
        else:  # info
            prob = min(1.0, size / denom)
            prob = max(prob, 1e-12)
            w = -math.log(prob)
        weights.append(float(w))
    return weights, float(sum(weights))


def lcs_length(pred: Sequence[str], ref_layers: Sequence[Set[str]]) -> int:
    """标准 LCS（pred[i] 命中 ref_layer[j] 的条件是 pred[i] in ref_layer[j]）"""
    n, m = len(pred), len(ref_layers)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        pi = pred[i - 1]
        for j in range(1, m + 1):
            if pi in ref_layers[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def weighted_lcs(pred: Sequence[str], ref_layers: Sequence[Set[str]], weights: Sequence[float]) -> float:
    """加权 LCS：命中第 j 层累加 weights[j]"""
    n, m = len(pred), len(ref_layers)
    if n == 0 or m == 0:
        return 0.0
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        pi = pred[i - 1]
        for j in range(1, m + 1):
            if pi in ref_layers[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + float(weights[j - 1])
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return float(dp[n][m])


def prediction_edges(pred: Sequence[str]) -> List[Tuple[str, str]]:
    """Prediction adjacent edges. Duplicates are kept for precision."""
    return [(pred[i], pred[i + 1]) for i in range(len(pred) - 1)]


def reference_edges(ref_layers: Sequence[Set[str]]) -> Set[Tuple[str, str]]:
    """Reference edges induced by Cartesian products between adjacent layers."""
    edges: Set[Tuple[str, str]] = set()
    for left, right in zip(ref_layers, ref_layers[1:]):
        for u in left:
            for v in right:
                edges.add((u, v))
    return edges


def edge_scores(pred: Sequence[str], ref_layers: Sequence[Set[str]]) -> Tuple[float, float, float]:
    """
    Edge Precision/Recall/F1 between prediction adjacent edges and reference-induced edges.

    Precision counts matched prediction edges over all prediction edges.
    Recall counts unique matched reference edges over all reference-induced edges.
    """
    pred_edges = prediction_edges(pred)
    ref_edge_set = reference_edges(ref_layers)
    if not pred_edges or not ref_edge_set:
        return 0.0, 0.0, 0.0

    matched_pred_count = sum(1 for edge in pred_edges if edge in ref_edge_set)
    matched_ref_edges = {edge for edge in pred_edges if edge in ref_edge_set}
    precision = matched_pred_count / len(pred_edges)
    recall = len(matched_ref_edges) / len(ref_edge_set)
    if precision + recall <= 0:
        return float(precision), float(recall), 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def harmonic_f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def score_candidate(
    pred: Sequence[str],
    ref: ReferenceEntry,
    lambda_penalty: float,
    vocab_size: int,
    use_chance_correct: bool,
) -> Dict[str, float]:
    """
    返回：LCS/WLCS 的 precision/recall/F1 及其 penalized score
    extra_steps = (|pred|-LCS) + (|ref|-LCS)
    """
    if not ref.layers:
        return {
            "lcs_len": 0.0,
            "wlcs": 0.0,
            "wlcsr_cc": 0.0,
            "lcsr": 0.0,
            "wlcsr": 0.0,
            "extra": 0.0,
            "penalty": 1.0,
            "lcsr_penalized": 0.0,
            "wlcsr_penalized": 0.0,
            "wlcsr_cc_penalized": 0.0,
            "lcs_precision": 0.0,
            "lcs_recall": 0.0,
            "lcs_f1": 0.0,
            "wlcs_precision": 0.0,
            "wlcs_recall": 0.0,
            "wlcs_f1": 0.0,
            "lcs_score": 0.0,
            "wlcs_score": 0.0,
            "edge_precision": 0.0,
            "edge_recall": 0.0,
            "edge_f1": 0.0,
            "sequence_score": 0.0,
            "strict_final_score": 0.0,
            "weighted_final_score": 0.0,
            "final_score": 0.0,
        }

    lcs_len = lcs_length(pred, ref.layers)
    if lcs_len == 1:
        lcs_len = 0
    wlcs = weighted_lcs(pred, ref.layers, ref.weights)
    if lcs_len == 0:   # 注意这里用 lcs_len（已经被截断过）
        wlcs = 0.0

    wlcsr_cc = 0.0
    if use_chance_correct:
        n = len(pred)
        V = max(int(vocab_size), 1)
        exp_w = 0.0
        for layer, w in zip(ref.layers, ref.weights):
            s = len(layer)
            p = min(1.0, s / V)
            q = 1.0 - (1.0 - p) ** n
            exp_w += w * q

        denom = ref.total_weight - exp_w
        if denom <= 1e-12:
            wlcsr_cc = 0.0
        else:
            wlcsr_cc = (wlcs - exp_w) / denom
        wlcsr_cc = max(0.0, min(1.0, wlcsr_cc))

    ref_len = len(ref.layers)
    lcsr = (lcs_len / ref_len) if ref_len > 0 else 0.0
    wlcsr = (wlcs / ref.total_weight) if ref.total_weight > 0 else 0.0

    extra = (len(pred) - lcs_len) + (len(ref.layers) - lcs_len)
    penalty = math.exp(-lambda_penalty * extra) if lambda_penalty > 0 else 1.0

    lcs_precision = (lcs_len / len(pred)) if pred else 0.0
    lcs_recall = lcsr
    lcs_f1 = harmonic_f1(lcs_precision, lcs_recall)
    lcs_score = float(lcs_f1 * penalty)

    avg_ref_weight = (ref.total_weight / ref_len) if ref_len > 0 else 0.0
    unmatched_pred_nodes = max(len(pred) - lcs_len, 0)
    wlcs_precision_denom = wlcs + unmatched_pred_nodes * avg_ref_weight
    wlcs_precision = (wlcs / wlcs_precision_denom) if wlcs_precision_denom > 1e-12 else 0.0
    wlcs_recall = wlcsr_cc if use_chance_correct else wlcsr
    wlcs_f1 = harmonic_f1(wlcs_precision, wlcs_recall)

    lcsr_penalized = float(lcsr * penalty)
    wlcsr_penalized = float(wlcsr * penalty)
    wlcsr_cc_penalized = float(wlcsr_cc * penalty if use_chance_correct else 0.0)
    wlcs_score = float(wlcs_f1 * penalty)
    edge_precision, edge_recall, edge_f1 = edge_scores(pred, ref.layers)
    sequence_score = (lcs_score + wlcs_score) / 2.0
    strict_final_score = (lcs_score + wlcs_score + edge_f1) / 3.0
    weighted_final_score = 0.45 * lcs_score + 0.45 * wlcs_score + 0.10 * edge_f1
    final_score = weighted_final_score

    return {
        "lcs_len": float(lcs_len),
        "wlcs": float(wlcs),
        "wlcsr_cc": float(wlcsr_cc),
        "lcsr": float(lcsr),
        "wlcsr": float(wlcsr),
        "extra": float(extra),
        "penalty": float(penalty),
        "lcsr_penalized": lcsr_penalized,
        "wlcsr_penalized": wlcsr_penalized,
        "wlcsr_cc_penalized": wlcsr_cc_penalized,
        "lcs_precision": float(lcs_precision),
        "lcs_recall": float(lcs_recall),
        "lcs_f1": float(lcs_f1),
        "wlcs_precision": float(wlcs_precision),
        "wlcs_recall": float(wlcs_recall),
        "wlcs_f1": float(wlcs_f1),
        "lcs_score": lcs_score,
        "wlcs_score": float(wlcs_score),
        "edge_precision": float(edge_precision),
        "edge_recall": float(edge_recall),
        "edge_f1": float(edge_f1),
        "sequence_score": float(sequence_score),
        "strict_final_score": float(strict_final_score),
        "weighted_final_score": float(weighted_final_score),
        "final_score": float(final_score),
    }


def load_reference_rows(
    reference_csv: str,
    reference_path_col: str,
    drop_epilepsy: bool,
    weight_mode: str,
    alpha: float,
    vocab_size_override: int,
) -> Tuple[List[ReferenceEntry], int]:
    df = pd.read_csv(reference_csv)

    if reference_path_col not in df.columns:
        raise ValueError(f"reference 缺少列：{reference_path_col}；实际列名：{list(df.columns)}")

    # 先收集 vocab（用于 info 权重）
    vocab: Set[str] = set()
    parsed_layers: List[Tuple[int, List[Set[str]]]] = []
    for idx, row in df.iterrows():
        layers = parse_reference_layers(row[reference_path_col], drop_epilepsy)
        if not layers:
            continue
        parsed_layers.append((int(idx), layers))
        for layer in layers:
            for node in layer:
                vocab.add(node)

    vocab_size = int(vocab_size_override) if vocab_size_override and vocab_size_override > 0 else max(len(vocab), 1)

    refs: List[ReferenceEntry] = []
    for row_idx, layers in parsed_layers:
        weights, total = layer_weights(layers, vocab_size, weight_mode, alpha)
        refs.append(
            ReferenceEntry(
                row_idx=row_idx,
                layers=layers,
                weights=weights,
                total_weight=total,
                repr_path=layers_to_repr(layers),
            )
        )

    if not refs:
        raise ValueError("reference 解析后为空：请检查 reference_path_col 与路径格式（是否包含 '->'）")

    return refs, vocab_size
