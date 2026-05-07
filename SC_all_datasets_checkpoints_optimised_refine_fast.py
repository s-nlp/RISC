"""
helpers_reranker.py

Utilities for:
- data preparation (parsing, normalization, leakage-safe filtering)
- candidate pruning / question filtering
- feature building with a feature switchboard (compute only requested feature sets)
- training LightGBM ranker
- inference over varying budgets
- saving/loading a "bundle" (model + config + feature list) for reuse

Key efficiency principles:
- Embedding cache shared across budgets (train) and across evaluation budgets (test).
- MiniLM feature builder supports `requested_features`: it only computes what you ask for
  (e.g., no chunk splitting if you didn't request chunk/step features).
- No duplicated function definitions: single source of truth.

Notes on leakage:
- All feature aggregation is done within each query id using only that query’s traces.
- No statistic is computed across train+test.
- Overlapping questions between train/test can be removed (recommended) via `remove_overlapping_questions`.
"""

import ast
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from sklearn.metrics.pairwise import cosine_similarity


import numpy as np
import pandas as pd

import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from math import comb
from itertools import product
# ============================================================
# Basic parsing / normalization
# ============================================================

def to_list_safe(x: Any) -> List[str]:
    """Parse a column that may contain a Python-list-like string."""
    if isinstance(x, list):
        return x
    try:
        v = ast.literal_eval(str(x))
        return v if isinstance(v, list) else []
    except Exception:
        return []


def normalize_answer_simple(ans: Any) -> str:
    s = str(ans)

    # Replace real newlines AND literal '\n' with spaces
    s = s.replace("\n", " ").replace("\\n", " ").replace("'\n '", " ")

    # Strip leading/trailing whitespace
    s = s.strip()

    # Strip surrounding quotes/brackets if they wrap the whole string
    s = s.strip('"\'' "[]")

    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)

    # Remove trailing junk like ."] or combinations of ., quotes, brackets (allow spaces)
    s = re.sub(r'[\s\'"\]]+$', '', s)
    s = re.sub(r'\.+$', '', s)

    # Final trim and lowercase
    return s.strip().lower()


def hit_substring(pred: Any, possible_list: Sequence[Any]) -> bool:
    """True if any possible answer is a substring of pred (case-insensitive)."""
    p = str(pred).lower()
    for a in possible_list:
        if str(a).lower() in p:
            return True
    return False


def compute_row_hit(
    df: pd.DataFrame,
    pred_col: str = "final_answer_new",
    possible_col: str = "possible_answers",
    out_possible_list_col: str = "possible_list",
    out_hit_col: str = "row_hit",
) -> pd.DataFrame:
    """
    Adds:
      - possible_list: parsed list
      - row_hit: bool
    """
    d = df.copy()
    d[out_possible_list_col] = d[possible_col].apply(to_list_safe)
    d[out_hit_col] = [
        hit_substring(p, lst) for p, lst in zip(d[pred_col].tolist(), d[out_possible_list_col].tolist())
    ]
    return d


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    ss = s.astype(str).str.strip().str.lower()
    return ss.isin(["true", "1", "t", "yes", "y"])


# ============================================================
# Leakage-safe filtering helpers
# ============================================================

def remove_overlapping_questions(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    question_col: str = "question",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes test rows whose question appears in train.
    Prevents "question leakage" when train and test are expanded from the same pool.
    """
    train_q = set(df_train[question_col].astype(str).tolist())
    dtest = df_test[~df_test[question_col].astype(str).isin(train_q)].copy()
    return df_train.copy(), dtest


def add_answer_id_within_id(
    df: pd.DataFrame,
    id_col: str = "id",
    answer_col: str = "final_answer_new",
    out_col: str = "answer_id",
) -> pd.DataFrame:
    """Assign integer answer_id within each query id by factorizing answer strings."""
    d = df.copy()
    d[out_col] = d.groupby(id_col)[answer_col].transform(lambda x: pd.factorize(x)[0]).astype(int)
    return d


# ============================================================
# Candidate pruning & question filtering
# ============================================================

def prune_answers_in_group(
    g: pd.DataFrame,
    a_col: str = "answer_id",
    min_answer_count: int = 1,
    keep_top_answers: int = 999999,
) -> pd.DataFrame:
    """
    Keep only rows whose answer_id appears >= min_answer_count within this id.
    If that drops everything, fallback to top keep_top_answers most frequent answers.
    Pruning ignores correctness label (uses ALL traces).
    """
    if min_answer_count <= 1:
        return g

    vc = g[a_col].astype(int).value_counts()
    keep = vc[vc >= min_answer_count].index
    if len(keep) > 0:
        g2 = g[g[a_col].astype(int).isin(keep)]
        if len(g2) > 0:
            return g2

    keep2 = vc.index[:keep_top_answers]
    g2 = g[g[a_col].astype(int).isin(keep2)]
    return g2 if len(g2) > 0 else g


def get_corr_ids_filtered(
    g: pd.DataFrame,
    a_col: str = "answer_id",
    hit_col: str = "row_hit",
    min_correct_count: int = 5,
    keep_top_correct: int = 2,
) -> np.ndarray:
    hit = to_bool_series(g[hit_col])
    cc = g.loc[hit, a_col].astype(int).value_counts()
    if cc.empty:
        return np.array([], dtype=int)
    corr_ids = cc[cc >= min_correct_count].index.to_numpy()
    if corr_ids.size == 0:
        corr_ids = cc.index[:keep_top_correct].to_numpy()
    return corr_ids.astype(int)


def filter_questions_for_reranker(
    df: pd.DataFrame,
    q_col: str = "id",
    a_col: str = "answer_id",
    hit_col: str = "row_hit",
    min_answer_count: int = 1,
    keep_top_answers: int = 100,
    use_filtered_corr: bool = False,
    min_correct_count: int = 5,
    keep_top_correct: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_kept_pruned: rows AFTER pruning, only for ids that pass all filters
      df_excl_pruned: rows AFTER pruning, for excluded ids
      per_q_stats: per-id stats (post-prune) with exclusion flags
    """
    kept_groups = []
    excl_groups = []
    rows = []

    for q, g in df.groupby(q_col, sort=False):
        g = g.copy()

        g = prune_answers_in_group(
            g, a_col=a_col,
            min_answer_count=min_answer_count,
            keep_top_answers=keep_top_answers,
        )

        hit = to_bool_series(g[hit_col])
        n_traces = int(len(g))
        n_answers = int(g[a_col].nunique())
        any_pos = bool(hit.any())

        if use_filtered_corr:
            corr_ids = get_corr_ids_filtered(
                g, a_col=a_col, hit_col=hit_col,
                min_correct_count=min_correct_count,
                keep_top_correct=keep_top_correct,
            )
            corr_set_size = int(len(corr_ids))
        else:
            corr_set_size = int(g.loc[hit, a_col].astype(int).nunique())

        ex_all_false = (not any_pos)
        ex_n_answers_le1 = (n_answers <= 1)
        ex_corr_eq_n_answers = (n_answers > 0 and corr_set_size == n_answers)

        exclude = ex_all_false or ex_n_answers_le1 or ex_corr_eq_n_answers
        # exclude = ex_all_false or ex_corr_eq_n_answers

        rows.append({
            q_col: int(q) if str(q).isdigit() else q,
            "n_traces": n_traces,
            "n_answers": n_answers,
            "any_pos": any_pos,
            "corr_set_size": corr_set_size,
            "ex_all_false": ex_all_false,
            "ex_n_answers_le1": ex_n_answers_le1,
            "ex_corr_eq_n_answers": ex_corr_eq_n_answers,
            "exclude": exclude,
        })

        if exclude:
            excl_groups.append(g)
        else:
            kept_groups.append(g)

    per_q_stats = pd.DataFrame(rows)
    df_kept_pruned = pd.concat(kept_groups, axis=0, ignore_index=True) if kept_groups else df.iloc[0:0].copy()
    df_excl_pruned = pd.concat(excl_groups, axis=0, ignore_index=True) if excl_groups else df.iloc[0:0].copy()
    return df_kept_pruned, df_excl_pruned, per_q_stats


# ============================================================
# Multi-budget expansion
# ============================================================

def make_multibudget_trace_df(
    df_trace: pd.DataFrame,
    budgets: Sequence[int],
    id_col: str = "id",
    call_col: Optional[str] = None,  # if None -> uses within-id row order
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expands each original query id into multiple "qid"s f"{orig_id}__{n}",
    where n is the budget (top_n calls kept).
    """
    budgets = sorted(set(int(b) for b in budgets))
    max_budget = max(budgets)

    d = df_trace.copy()
    if call_col is None:
        d["_call_idx"] = d.groupby(id_col, sort=False).cumcount() + 1
        t_col = "_call_idx"
    else:
        d[call_col] = pd.to_numeric(d[call_col], errors="coerce")
        d = d.dropna(subset=[call_col])
        t_col = call_col

    d = d.sort_values([id_col, t_col], kind="mergesort")

    parts = []
    meta_rows = []

    for n in budgets:
        prefix = d.groupby(id_col, sort=False, group_keys=False).head(n).copy()
        prefix["orig_id"] = prefix[id_col]
        prefix["budget_n"] = n
        prefix["budget_n_norm"] = n / max_budget
        prefix[id_col] = prefix["orig_id"].astype(str) + "__" + str(n)
        parts.append(prefix)
        meta_rows.append(prefix[[id_col, "orig_id", "budget_n", "budget_n_norm"]].drop_duplicates(subset=[id_col]))

    df_mb_trace = pd.concat(parts, ignore_index=True)
    qid_meta = pd.concat(meta_rows, ignore_index=True).drop_duplicates(subset=[id_col])
    return df_mb_trace, qid_meta


# ============================================================
# Feature definitions
# ============================================================

HANDCRAFTED_FEATURES_ALL = [
    "ans_rows", "rat_len_mean", "rat_len_min", "rat_len_max",
    "ans_len_mean", "ans_len_min", "ans_len_max", "rat_to_ans_len",
    "ans_rows_rel", "ans_len_min_rel", "rat_len_max_rel", "rat_to_ans_len_rel",
    "ans_share", "top1_share", "share_margin_to_best", "share_ratio_to_best", "entropy_norm", "hhi",
    "last_seen_call", "streak_max_norm", "max_plateau_len_norm", "max_plateau_len",
    "max_internal_gap", "first_gap", "tail_plateau_len",
]

# Keep as a "traditional base"; you can pass custom embedding_cols to FeatureConfig.
EMBED_FEATURES_BASE = [
    "cot_dispersion_id_answer",
    "ans_dist2_to_id_centroid",
    "ans_dist2_to_top1_emb",
    "cot_ans_cos_min",
    "cot_centroid_dist_to_id_cot_centroid",
    "cot_centroid_dist_to_top1_cot_centroid",
]

# Extended features supported by the switchboard MiniLM builder.
EMBED_FEATURES_MINILM_ALL = [
    # Answer-geometry
    "ans_dist2_to_id_centroid",
    "ans_dist2_to_top1_emb",

    # CoT full-text
    "cot_dispersion_id_answer",
    "cot_centroid_dist_to_id_cot_centroid",
    "cot_centroid_dist_to_top1_cot_centroid",
    "cot_ans_cos_min",
    "cot_pairwise_l2_max",

    # Question relevance
    "qa_cos",
    "qa_dist2",

    # Q ↔ CoT (full CoT)
    "qc_cos_min",
    "qc_cos_max",
    "qc_cos_range",

    # Chunk/call
    "cot_chunk_disp_mean",
    "cot_call_dispersion_in_answer",
    "cot_centroid_dist_to_id_cot_centroid_chunk",
    "step_prev_repeat_frac",

    # Last step
    "last_step_dispersion",
    "last_step_dispersion_min",
    "last_step_dispersion_max",
    "last_step_to_answer_cos",
    "last_step_to_answer_cos_max",

    # Step coherence
    "step_adj_cos_min",
    "step_adj_cos_std",
    "step_adj_cos_last2",
    "step_unique_step_ratio",
    "step_to_chain_centroid_mean",
    "step_to_chain_centroid_min",

    # Q × step semantics
    "last_step_to_q_cos_mean",
    "last_step_to_q_cos_min",
    "q_step_cos_min_call_mean",
    "q_step_cos_max_call_mean",
    "q_step_cos_range_call_mean",
    "q_step_cos_last2_mean",
    "q_step_cos_last2_min",
]


# ============================================================
# Prefix-convergence feature names
# ============================================================

PREFIX_CONVERGENCE_FEATURES_MINILM_ALL = [
    "prefix_n_traces",
    "prefix_n_prefix_points",
    "prefix_converge_frac",
    "prefix_converge_depth_mean",
    "prefix_converge_depth_min",
    "shared_checkpoint_count",
    "shared_checkpoint_frac",
    "shared_checkpoint_per_trace",
    "max_checkpoint_coverage",
    "mean_checkpoint_coverage",
    "early_checkpoint_coverage",
    "late_checkpoint_coverage",
    "matched_trace_pair_frac",
    "mean_prefixes_per_trace",
]


# ============================================================
# Separate caches for base embeddings and prefix embeddings
# ============================================================

@dataclass
class EmbeddingCaches:
    """
    Separate persistent CPU embedding caches plus non-persistent runtime artifacts.
      - base_cache: raw/text embeddings for add_id_answer_embedding_features_minilm
      - prefix_cache: raw/text embeddings for add_id_answer_prefix_convergence_features_minilm
      - runtime_cache: derived per-dataframe artifacts reused across budgets within one run
    """
    base_cache: Optional[Dict] = None
    prefix_cache: Optional[Dict] = None
    runtime_cache: Optional[Dict] = None

    def __post_init__(self):
        if self.base_cache is None:
            self.base_cache = {}
        if self.prefix_cache is None:
            self.prefix_cache = {}
        if self.runtime_cache is None:
            self.runtime_cache = {}

    def clear_runtime(self) -> None:
        self.runtime_cache = {}

    def save(self, path: str | Path) -> None:
        import joblib
        payload = {
            "base_cache": self.base_cache,
            "prefix_cache": self.prefix_cache,
        }
        joblib.dump(payload, str(path))

    @staticmethod
    def load(path: str | Path) -> "EmbeddingCaches":
        import joblib
        payload = joblib.load(str(path))
        return EmbeddingCaches(
            base_cache=dict(payload.get("base_cache", {})),
            prefix_cache=dict(payload.get("prefix_cache", {})),
            runtime_cache={},
        )


# ------------------------------------------------------------------
# Ephemeral GPU cache registry:
#   - persistent CPU caches stay in EmbeddingCaches.{base,prefix}_cache
#   - GPU cache lives only for the current process/run
#   - keyed by id(embed_cache) so existing call sites do not need to change
# ------------------------------------------------------------------
_EMBED_GPU_CACHE_REGISTRY: Dict[Tuple[int, str], Dict[Any, torch.Tensor]] = {}


def _get_ephemeral_gpu_cache(embed_cache: Dict, device: str) -> Dict[Any, torch.Tensor]:
    key = (id(embed_cache), str(device))
    cache = _EMBED_GPU_CACHE_REGISTRY.get(key)
    if cache is None:
        cache = {}
        _EMBED_GPU_CACHE_REGISTRY[key] = cache
    return cache


def clear_ephemeral_gpu_cache() -> None:
    _EMBED_GPU_CACHE_REGISTRY.clear()

#adaptive set1
def choose_checkpoint_thresholds(top_n: int) -> tuple[float, float]:
    if top_n <= 15:
        return 0.75, 0.50
    elif top_n <= 40:
        return 0.80, 0.50
    elif top_n <= 70:
        return 0.88, 0.50
    else:
        return 0.90, 0.25



    

def _resolve_prefix_thresholds(feature_cfg: "FeatureConfig", top_n: int) -> Tuple[float, float]:
    if getattr(feature_cfg, "adaptive_prefix_thresholds", False):
        return choose_checkpoint_thresholds(int(top_n))
    return float(feature_cfg.prefix_sim_threshold), float(feature_cfg.prefix_rel_support_thr)


# ============================================================
# Handcrafted feature builder (fast, selected)
# ============================================================

def build_id_answer_features_selected(
    df: pd.DataFrame,
    top_n: int,
    feature_cols: Sequence[str],
    id_col: str = "id",
    answer_col: str = "final_answer_new",
    hit_col: str = "row_hit",
    rationale_col: str = "rationale",
    call_col: Optional[str] = None,
    sort_cols: Optional[Sequence[str]] = None,
    ascending: bool = True,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Builds only requested handcrafted features + [id, answer, hit_mean].
    """
    need = [id_col, answer_col, hit_col, rationale_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df.copy()
    d[hit_col] = pd.to_numeric(d[hit_col], errors="coerce").fillna(0.0)

    d["rationale_len"] = d[rationale_col].fillna("").astype(str).str.len()
    d["answer_len"] = d[answer_col].fillna("").astype(str).str.len()

    if sort_cols is not None:
        d = d.sort_values([id_col] + list(sort_cols),
                          ascending=[True] + [ascending] * len(sort_cols),
                          kind="mergesort")
    else:
        d = d.sort_values([id_col], kind="mergesort")

    if call_col is None:
        d["_call_idx"] = d.groupby(id_col, sort=False).cumcount() + 1
        t_col = "_call_idx"
    else:
        d[call_col] = pd.to_numeric(d[call_col], errors="coerce")
        d = d.dropna(subset=[call_col])
        t_col = call_col

    d = d.sort_values([id_col, t_col], kind="mergesort")
    d_top = d.groupby(id_col, sort=False, group_keys=False).head(top_n).copy()

    feats = (
        d_top
        .groupby([id_col, answer_col], as_index=False, dropna=False)
        .agg(
            hit_mean=(hit_col, "mean"),
            ans_rows=(answer_col, "size"),
            rat_len_mean=("rationale_len", "mean"),
            rat_len_min=("rationale_len", "min"),
            rat_len_max=("rationale_len", "max"),
            ans_len_mean=("answer_len", "mean"),
            ans_len_min=("answer_len", "min"),
            ans_len_max=("answer_len", "max"),
        )
    )

    feats["rat_to_ans_len"] = feats["rat_len_mean"] / (feats["ans_len_mean"] + 1.0)

    rel_base = ["ans_rows", "ans_len_min", "rat_len_max", "rat_to_ans_len"]
    for c in rel_base:
        feats[c + "_rel"] = feats[c] - feats.groupby(id_col, sort=False)[c].transform("min")

    # competitiveness
    total_rows = feats.groupby(id_col, sort=False)["ans_rows"].transform("sum")
    feats["ans_share"] = feats["ans_rows"] / (total_rows + eps)
    top1_share = feats.groupby(id_col, sort=False)["ans_share"].transform("max")
    feats["top1_share"] = top1_share
    feats["share_margin_to_best"] = top1_share - feats["ans_share"]
    feats["share_ratio_to_best"] = feats["ans_share"] / (top1_share + eps)

    # entropy/hhi
    K = feats.groupby(id_col, sort=False)["ans_share"].transform("count").astype(float)
    p = feats["ans_share"].to_numpy()
    p_logp = np.where(p > 0, p * np.log(p), 0.0)
    feats["_p_logp"] = p_logp
    ent = -feats.groupby(id_col, sort=False)["_p_logp"].transform("sum")
    feats["entropy_norm"] = np.where(K > 1, ent / (np.log(K) + eps), 0.0)
    feats["_p2"] = feats["ans_share"] ** 2
    feats["hhi"] = feats.groupby(id_col, sort=False)["_p2"].transform("sum")
    feats.drop(columns=["_p_logp", "_p2"], inplace=True)

    # dynamics only if requested
    dyn_wanted = set(feature_cols).intersection({
        "last_seen_call", "streak_max_norm", "max_plateau_len_norm", "max_plateau_len",
        "max_internal_gap", "first_gap", "tail_plateau_len"
    })
    if dyn_wanted:
        g = d_top.groupby([id_col, answer_col], sort=False, dropna=False)
        last_seen = g[t_col].max().rename("last_seen_call").reset_index()

        prev_id = d_top[id_col].shift(1)
        prev_ans = d_top[answer_col].shift(1)
        new_run = (d_top[id_col] != prev_id) | (d_top[answer_col] != prev_ans)
        d_top["_run_id"] = new_run.cumsum()

        run_len = d_top.groupby("_run_id", sort=False).size().rename("run_len")
        run_keys = d_top.groupby("_run_id", sort=False).agg(
            **{id_col: (id_col, "first"), answer_col: (answer_col, "first")}
        )
        runs = run_keys.join(run_len).reset_index(drop=True)

        streak_max = (
            runs.groupby([id_col, answer_col], sort=False, dropna=False)["run_len"]
            .max().rename("streak_max").reset_index()
        )
        streak_max["streak_max_norm"] = streak_max["streak_max"] / float(top_n)

        d_pos = d_top[[id_col, answer_col, t_col]].copy()
        d_pos["_prev_t"] = d_pos.groupby([id_col, answer_col], sort=False, dropna=False)[t_col].shift(1)
        d_pos["_gap"] = d_pos[t_col] - d_pos["_prev_t"] - 1
        internal_max_gap = (
            d_pos.groupby([id_col, answer_col], sort=False, dropna=False)["_gap"]
            .max().rename("max_internal_gap").reset_index()
        )
        first_seen = g[t_col].min().rename("first_seen_call").reset_index()
        first_gap = first_seen.copy()
        first_gap["first_gap"] = first_gap["first_seen_call"] - 1
        tail_gap = last_seen.copy()
        tail_gap["tail_plateau_len"] = top_n - tail_gap["last_seen_call"]

        dyn = (
            last_seen
            .merge(streak_max[[id_col, answer_col, "streak_max_norm"]], on=[id_col, answer_col], how="left")
            .merge(internal_max_gap, on=[id_col, answer_col], how="left")
            .merge(first_gap[[id_col, answer_col, "first_gap"]], on=[id_col, answer_col], how="left")
            .merge(tail_gap[[id_col, answer_col, "tail_plateau_len"]], on=[id_col, answer_col], how="left")
        )
        dyn["max_internal_gap"] = dyn["max_internal_gap"].fillna(0.0)
        dyn["first_gap"] = dyn["first_gap"].fillna(0.0)
        dyn["tail_plateau_len"] = dyn["tail_plateau_len"].fillna(top_n)
        dyn["max_plateau_len"] = dyn[["max_internal_gap", "first_gap", "tail_plateau_len"]].max(axis=1)
        dyn["max_plateau_len_norm"] = dyn["max_plateau_len"] / float(top_n)

        dyn_keep = [id_col, answer_col] + [c for c in dyn.columns if c in dyn_wanted]
        feats = feats.merge(dyn[dyn_keep], on=[id_col, answer_col], how="left")
        for c in dyn_wanted:
            feats[c] = feats[c].fillna(0.0)

    keep_cols = [id_col, answer_col, "hit_mean"] + list(feature_cols)
    missing_feats = [c for c in feature_cols if c not in feats.columns]
    if missing_feats:
        raise ValueError(f"Requested feature_cols missing from output: {missing_feats}")
    return feats[keep_cols].copy()


# ============================================================
# MiniLM embedding feature builder (SWITCHBOARD, efficient)
# ============================================================

def add_id_answer_embedding_features_minilm(
    df: pd.DataFrame,
    top_n: int,
    id_col: str = "id",
    question_col: str = "question",
    answer_col: str = "final_answer_new",
    rationale_col: str = "rationale",
    call_col: Optional[str] = None,
    sort_cols: Optional[Sequence[str]] = None,
    ascending: bool = True,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
    device: str = "cuda",
    normalize_embeddings: bool = True,
    answer_centroid_weighted_by_calls: bool = True,
    embed_cache: Optional[Dict] = None,
    max_chunks_per_cot: int = 64,
    min_chunk_chars: int = 3,
    step_unique_thr: float = 0.92,
    step_prev_repeat_gap: int = 2,
    step_prev_repeat_thr: float = 0.8,
    requested_features: Optional[Sequence[str]] = None,
    
) -> pd.DataFrame:
    """
    Efficient embedding feature builder.
    Computes only the columns in `requested_features` (plus id/answer).
    """
    import numpy as _np

    need = [id_col, answer_col, rationale_col]
    if requested_features is None:
        requested = set(EMBED_FEATURES_MINILM_ALL)
    else:
        requested = set(requested_features)

    # Validate required columns conditionally (question only if needed)
    needs_q = any(c in requested for c in [
        "qa_cos", "qa_dist2", "qc_cos_min", "qc_cos_max", "qc_cos_range",
        "last_step_to_q_cos_mean", "last_step_to_q_cos_min",
        "q_step_cos_min_call_mean", "q_step_cos_max_call_mean",
        "q_step_cos_range_call_mean", "q_step_cos_last2_mean", "q_step_cos_last2_min", "cot_pairwise_l2_max"
    ])
    if needs_q:
        need.append(question_col)

    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if embed_cache is None:
        embed_cache = {}

    # Fast flags
    need_U = any(f in requested for f in [
    "cot_dispersion_id_answer", "cot_pairwise_l2_max",
    "cot_centroid_dist_to_id_cot_centroid",
    "cot_centroid_dist_to_top1_cot_centroid", "cot_ans_cos_min",
    "qc_cos_min", "qc_cos_max", "qc_cos_range",])
    
    need_V = any(f in requested for f in [
        "ans_dist2_to_id_centroid", "ans_dist2_to_top1_emb",
        "qa_cos", "qa_dist2",
        "cot_ans_cos_min",
        "last_step_to_answer_cos", "last_step_to_answer_cos_max",
    ])
    need_Q = any(f in requested for f in [
        "qa_cos", "qa_dist2",
        "qc_cos_min", "qc_cos_max", "qc_cos_range",
        "last_step_to_q_cos_mean", "last_step_to_q_cos_min",
        "q_step_cos_min_call_mean", "q_step_cos_max_call_mean",
        "q_step_cos_range_call_mean", "q_step_cos_last2_mean", "q_step_cos_last2_min",
    ])
    need_chunks = any(f in requested for f in [
        "cot_chunk_disp_mean", "cot_call_dispersion_in_answer",
        "cot_centroid_dist_to_id_cot_centroid_chunk",
        "last_step_dispersion", "last_step_dispersion_min", "last_step_dispersion_max",
        "last_step_to_answer_cos", "last_step_to_answer_cos_max",
        "step_adj_cos_min", "step_adj_cos_std", "step_adj_cos_last2",
        "step_unique_step_ratio", "step_to_chain_centroid_mean", "step_to_chain_centroid_min",
        "last_step_to_q_cos_mean", "last_step_to_q_cos_min",
        "q_step_cos_min_call_mean", "q_step_cos_max_call_mean",
        "q_step_cos_range_call_mean", "q_step_cos_last2_mean", "q_step_cos_last2_min", "step_prev_repeat_frac",
    ])

    # ---- sort + take top_n per id
    d = df.copy()
    if sort_cols is not None:
        d = d.sort_values([id_col] + list(sort_cols),
                          ascending=[True] + [ascending] * len(sort_cols),
                          kind="mergesort")
    else:
        d = d.sort_values([id_col], kind="mergesort")

    if call_col is None:
        d["_call_idx"] = d.groupby(id_col, sort=False).cumcount() + 1
        t_col = "_call_idx"
    else:
        d[call_col] = pd.to_numeric(d[call_col], errors="coerce")
        d = d.dropna(subset=[call_col])
        t_col = call_col

    d = d.sort_values([id_col, t_col], kind="mergesort")
    d_top = d.groupby(id_col, sort=False, group_keys=False).head(top_n).copy()

    d_top["_q_text"] = d_top[question_col].fillna("").astype(str) if needs_q else ""
    d_top["_ans_text"] = d_top[answer_col].fillna("").astype(str)
    d_top["_cot_text"] = d_top[rationale_col].fillna("").astype(str)

    uniq = (
        d_top.groupby([id_col, "_ans_text"], as_index=False, dropna=False)
        .size()
        .rename(columns={"size": "ans_calls", "_ans_text": answer_col})
    )
    uniq["pair_idx"] = _np.arange(len(uniq), dtype=_np.int64)

    id_to_idx: Dict[Any, int] = {}
    ids_in_order: List[Any] = []
    for _id in uniq[id_col].tolist():
        if _id not in id_to_idx:
            id_to_idx[_id] = len(id_to_idx)
            ids_in_order.append(_id)
    uniq["id_idx"] = uniq[id_col].map(id_to_idx).astype(_np.int64)

    d_top = d_top.merge(
        uniq[[id_col, answer_col, "pair_idx", "id_idx"]],
        left_on=[id_col, "_ans_text"],
        right_on=[id_col, answer_col],
        how="left",
        sort=False,
    )
    d_top.rename(columns={"_ans_text": answer_col}, inplace=True)

    model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def _encode_with_cache(texts: List[str], kind: str) -> torch.Tensor:
        texts = ["" if t is None else str(t) for t in texts]
        if len(texts) == 0:
            return torch.empty((0, 0), dtype=torch.float32, device=device)

        gpu_cache = _get_ephemeral_gpu_cache(embed_cache, device)
        keys = [(kind, t) for t in texts]

        out_chunks: List[Optional[torch.Tensor]] = [None] * len(texts)
        miss_idx: List[int] = []
        miss_texts: List[str] = []

        for i, k in enumerate(keys):
            g = gpu_cache.get(k)
            if g is not None:
                out_chunks[i] = g
                continue

            cpu_val = embed_cache.get(k)
            if cpu_val is not None:
                t = torch.from_numpy(cpu_val).to(device=device, dtype=torch.float32, non_blocking=True)
                gpu_cache[k] = t
                out_chunks[i] = t
            else:
                miss_idx.append(i)
                miss_texts.append(texts[i])

        if miss_idx:
            miss_emb = model.encode(
                miss_texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False,
            )
            if miss_emb.dtype != torch.float32:
                miss_emb = miss_emb.float()

            miss_emb_cpu = miss_emb.detach().cpu().numpy().astype(np.float32, copy=False)
            for j, i in enumerate(miss_idx):
                k = keys[i]
                embed_cache[k] = miss_emb_cpu[j]
                gpu_cache[k] = miss_emb[j]
                out_chunks[i] = miss_emb[j]

        return torch.stack(out_chunks, dim=0)

    @torch.no_grad()
    def group_stats(emb: torch.Tensor, gidx: torch.Tensor, G: int):
        """Return (disp[G], cent[G,D]) where disp is mean squared distance to centroid."""
        if emb.dtype != torch.float32:
            emb = emb.float()
        dev = emb.device
        gidx = gidx.to(dev)

        counts = torch.bincount(gidx, minlength=G).float().to(dev)
        D = emb.shape[1]
        sums = torch.zeros((G, D), device=dev, dtype=torch.float32)
        sums.index_add_(0, gidx, emb)
        cent = sums / counts.clamp_min(1.0).unsqueeze(1)

        x_norm2 = (emb * emb).sum(dim=1)
        sum_norm2 = torch.zeros((G,), device=dev, dtype=torch.float32)
        sum_norm2.index_add_(0, gidx, x_norm2)
        e_norm2 = sum_norm2 / counts.clamp_min(1.0)

        cent_norm2 = (cent * cent).sum(dim=1)
        disp = (e_norm2 - cent_norm2).clamp_min(0.0)
        return disp, cent

    # Chunking by "."
    _dot_split = re.compile(r"\.(?=\s|$)")

    def split_by_dot(text: str) -> List[str]:
        t = (text or "").strip()
        if not t:
            return []
        t = re.sub(r"\s+", " ", t)
        parts = _dot_split.split(t)
        chunks: List[str] = []
        for p in parts:
            s = p.strip()
            if not s:
                continue
            if not s.endswith("."):
                s += "."
            if len(s) < min_chunk_chars:
                continue
            chunks.append(s)
            if len(chunks) >= max_chunks_per_cot:
                break
        if not chunks:
            chunks = [t[:4000]]
        return chunks

    # Encode as needed
    U = V = Q_id = None
    if need_U:
        U = _encode_with_cache(d_top["_cot_text"].tolist(), kind="cot")  # [N,D]
    if need_V:
        V = _encode_with_cache(uniq[answer_col].fillna("").astype(str).tolist(), kind="ans")  # [M,D]
    if need_Q:
        q_by_id = (
            d_top[[id_col, "_q_text"]]
            .drop_duplicates(subset=[id_col], keep="first")
            .set_index(id_col)["_q_text"]
            .reindex(ids_in_order)
            .fillna("")
            .astype(str)
            .tolist()
        )
        Q_id = _encode_with_cache(q_by_id, kind="q")  # [G_id,D]

    dev = (U.device if U is not None else (V.device if V is not None else Q_id.device))
    pair_idx_t = torch.from_numpy(d_top["pair_idx"].to_numpy(np.int64)).to(dev)
    id_idx_call_t = torch.from_numpy(d_top["id_idx"].to_numpy(np.int64)).to(dev)
    id_idx_uniq_t = torch.from_numpy(uniq["id_idx"].to_numpy(np.int64)).to(dev)

    N = len(d_top)
    M = len(uniq)
    G_id = len(ids_in_order)

    # Top1 mapping if needed
    need_top1 = any(f in requested for f in ["ans_dist2_to_top1_emb", "cot_centroid_dist_to_top1_cot_centroid"])
    if need_top1:
        top1_rows = (
            uniq.sort_values([id_col, "ans_calls", "pair_idx"], ascending=[True, False, True], kind="mergesort")
            .groupby(id_col, sort=False, as_index=False)
            .head(1)[["id_idx", "pair_idx"]]
        )
        top1_pair_idx_per_id = np.full((G_id,), -1, dtype=np.int64)
        top1_pair_idx_per_id[top1_rows["id_idx"].to_numpy(np.int64)] = top1_rows["pair_idx"].to_numpy(np.int64)
        top1_pair_idx_t = torch.from_numpy(top1_pair_idx_per_id).to(dev)
    else:
        top1_pair_idx_t = None

    out = uniq[[id_col, answer_col]].copy()

    # Answer geometry
    if any(f in requested for f in ["ans_dist2_to_id_centroid", "ans_dist2_to_top1_emb"]):
        weights_unique = uniq["ans_calls"].to_numpy(np.float32) if answer_centroid_weighted_by_calls else None
        w_u = torch.ones((M,), device=dev, dtype=torch.float32) if weights_unique is None else torch.from_numpy(weights_unique).to(dev)

        D = V.shape[1]
        sums = torch.zeros((G_id, D), device=dev, dtype=torch.float32)
        sums.index_add_(0, id_idx_uniq_t, V * w_u.unsqueeze(1))
        wsum = torch.zeros((G_id,), device=dev, dtype=torch.float32)
        wsum.index_add_(0, id_idx_uniq_t, w_u)
        ans_cent_id = sums / wsum.clamp_min(1.0).unsqueeze(1)

        if "ans_dist2_to_id_centroid" in requested:
            out["ans_dist2_to_id_centroid"] = ((V - ans_cent_id[id_idx_uniq_t]) ** 2).sum(dim=1).detach().cpu().numpy()
        if "ans_dist2_to_top1_emb" in requested:
            top1_ans_emb_for_row = V[top1_pair_idx_t[id_idx_uniq_t]]
            out["ans_dist2_to_top1_emb"] = ((V - top1_ans_emb_for_row) ** 2).sum(dim=1).detach().cpu().numpy()

    # Full-CoT
    if any(f in requested for f in [
        "cot_dispersion_id_answer",
        "cot_pairwise_l2_max",
        "cot_centroid_dist_to_id_cot_centroid",
        "cot_centroid_dist_to_top1_cot_centroid",
    ]):
        cot_disp, cot_cent_pair = group_stats(U, pair_idx_t, M)
    
        if "cot_dispersion_id_answer" in requested:
            out["cot_dispersion_id_answer"] = cot_disp.detach().cpu().numpy()
    
        if "cot_pairwise_l2_max" in requested:
            pair_idx_np = d_top["pair_idx"].to_numpy(np.int64)
            max_l2 = np.zeros((M,), dtype=np.float32)
    
            # compute per answer cluster
            for pidx, gtmp in d_top.groupby("pair_idx", sort=False):
                idxs = gtmp.index.to_numpy()
                if len(idxs) <= 1:
                    max_l2[pidx] = 0.0
                    continue
    
                X = U[idxs].detach().cpu().numpy().astype(np.float32, copy=False)  # [k, D]
    
                # normalized embeddings => squared L2 = 2 - 2*cos
                S = X @ X.T
                D2 = np.maximum(0.0, 2.0 - 2.0 * S)
                D = np.sqrt(D2, dtype=np.float32)
                max_l2[pidx] = float(D.max())
    
            out["cot_pairwise_l2_max"] = max_l2
    
        if any(f in requested for f in ["cot_centroid_dist_to_id_cot_centroid", "cot_centroid_dist_to_top1_cot_centroid"]):
            _, cot_cent_id = group_stats(U, id_idx_call_t, G_id)
            cot_cent_id_for_pair = cot_cent_id[id_idx_uniq_t]
    
            if "cot_centroid_dist_to_id_cot_centroid" in requested:
                out["cot_centroid_dist_to_id_cot_centroid"] = (
                    ((cot_cent_pair - cot_cent_id_for_pair) ** 2)
                    .sum(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
    if "cot_ans_cos_min" in requested:
        v_call = V[pair_idx_t]
        cos_call = (U * v_call).sum(dim=1).detach().cpu().numpy()
        pair_idx_np = d_top["pair_idx"].to_numpy(np.int64)
        tmp = pd.DataFrame({"pair_idx": pair_idx_np, "cos": cos_call})
        out["cot_ans_cos_min"] = (
            tmp.groupby("pair_idx", sort=False)["cos"].min()
            .reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
        )

    # Q–A
    if any(f in requested for f in ["qa_cos", "qa_dist2"]):
        q_for_pair = Q_id[id_idx_uniq_t]
        if "qa_cos" in requested:
            out["qa_cos"] = (q_for_pair * V).sum(dim=1).detach().cpu().numpy()
        if "qa_dist2" in requested:
            out["qa_dist2"] = ((q_for_pair - V) ** 2).sum(dim=1).detach().cpu().numpy()

    # Q–CoT (full CoT)
    if any(f in requested for f in ["qc_cos_min", "qc_cos_max", "qc_cos_range"]):
        q_for_call = Q_id[id_idx_call_t]            # [N,D]
        qc_cos_call = (q_for_call * U).sum(dim=1).detach().cpu().numpy()
        pair_idx_np = d_top["pair_idx"].to_numpy(np.int64)
        tmp_qc = pd.DataFrame({"pair_idx": pair_idx_np, "qc": qc_cos_call})
        if "qc_cos_min" in requested:
            out["qc_cos_min"] = tmp_qc.groupby("pair_idx", sort=False)["qc"].min().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
        if "qc_cos_max" in requested:
            out["qc_cos_max"] = tmp_qc.groupby("pair_idx", sort=False)["qc"].max().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
        if "qc_cos_range" in requested:
            qc_max = tmp_qc.groupby("pair_idx", sort=False)["qc"].max().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
            qc_min = tmp_qc.groupby("pair_idx", sort=False)["qc"].min().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
            out["qc_cos_range"] = qc_max - qc_min

    # Chunk/call/step + Q×step
    if need_chunks:
        chunk_texts: List[str] = []
        chunk_call_idx: List[int] = []
        last_chunk_texts: List[str] = []

        cot_texts = d_top["_cot_text"].tolist()
        for i, txt in enumerate(cot_texts):
            chunks = split_by_dot(txt)
            if not chunks:
                chunks = [""]
            last_chunk_texts.append(chunks[-1])
            for ch in chunks:
                chunk_texts.append(ch)
                chunk_call_idx.append(i)

        E_chunks = _encode_with_cache(chunk_texts, kind="cot_chunk")          # [Nc,D]
        E_last = _encode_with_cache(last_chunk_texts, kind="cot_chunk")       # [N,D]
        call_idx_t = torch.from_numpy(np.asarray(chunk_call_idx, dtype=np.int64)).to(dev)

        cot_chunk_disp_call_t, call_emb_t = group_stats(E_chunks, call_idx_t, N)
        cot_call_disp_t, call_emb_pair_cent_t = group_stats(call_emb_t, pair_idx_t, M)

        if "cot_chunk_disp_mean" in requested:
            pair_idx_np = d_top["pair_idx"].to_numpy(np.int64)
            tmp = pd.DataFrame({"pair_idx": pair_idx_np, "cdisp": cot_chunk_disp_call_t.detach().cpu().numpy()})
            out["cot_chunk_disp_mean"] = tmp.groupby("pair_idx", sort=False)["cdisp"].mean().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)

        if "cot_call_dispersion_in_answer" in requested:
            out["cot_call_dispersion_in_answer"] = cot_call_disp_t.detach().cpu().numpy()

        if "cot_centroid_dist_to_id_cot_centroid_chunk" in requested:
            _, call_emb_id_cent_t = group_stats(call_emb_t, id_idx_call_t, G_id)
            call_emb_id_for_pair_t = call_emb_id_cent_t[id_idx_uniq_t]
            out["cot_centroid_dist_to_id_cot_centroid_chunk"] = ((call_emb_pair_cent_t - call_emb_id_for_pair_t) ** 2).sum(dim=1).detach().cpu().numpy()

        if any(f in requested for f in ["last_step_dispersion", "last_step_dispersion_min", "last_step_dispersion_max"]):
            last_step_disp_t, last_pair_cent_t = group_stats(E_last, pair_idx_t, M)
            if "last_step_dispersion" in requested:
                out["last_step_dispersion"] = last_step_disp_t.detach().cpu().numpy()
            if any(f in requested for f in ["last_step_dispersion_min", "last_step_dispersion_max"]):
                last_cent_for_call = last_pair_cent_t[pair_idx_t]
                last_dist2_call = ((E_last - last_cent_for_call) ** 2).sum(dim=1).detach().cpu().numpy()
                pair_idx_np = d_top["pair_idx"].to_numpy(np.int64)
                tmp = pd.DataFrame({"pair_idx": pair_idx_np, "d2": last_dist2_call})
                if "last_step_dispersion_min" in requested:
                    out["last_step_dispersion_min"] = tmp.groupby("pair_idx", sort=False)["d2"].min().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
                if "last_step_dispersion_max" in requested:
                    out["last_step_dispersion_max"] = tmp.groupby("pair_idx", sort=False)["d2"].max().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)

        if any(f in requested for f in ["last_step_to_answer_cos", "last_step_to_answer_cos_max"]):
            v_call = V[pair_idx_t]
            l2a = (E_last * v_call).sum(dim=1).detach().cpu().numpy()
            pair_idx_np = d_top["pair_idx"].to_numpy(np.int64)
            tmp = pd.DataFrame({"pair_idx": pair_idx_np, "l2a": l2a})
            if "last_step_to_answer_cos" in requested:
                out["last_step_to_answer_cos"] = tmp.groupby("pair_idx", sort=False)["l2a"].mean().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
            if "last_step_to_answer_cos_max" in requested:
                out["last_step_to_answer_cos_max"] = tmp.groupby("pair_idx", sort=False)["l2a"].max().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)

        # Step coherence
        need_step = any(f in requested for f in [
            "step_adj_cos_min", "step_adj_cos_std", "step_adj_cos_last2",
            "step_unique_step_ratio", "step_to_chain_centroid_mean", "step_to_chain_centroid_min", "step_prev_repeat_frac"
        ])
        if need_step:
            same_call = (call_idx_t[1:] == call_idx_t[:-1])
            adj_cos_t = (E_chunks[:-1] * E_chunks[1:]).sum(dim=1)
            adj_cos_t = adj_cos_t[same_call]
            adj_call_idx_t = call_idx_t[:-1][same_call]

            adj_cos = adj_cos_t.detach().cpu().numpy().astype(np.float32, copy=False)
            adj_call_idx = adj_call_idx_t.detach().cpu().numpy().astype(np.int64, copy=False)
            adj_df = pd.DataFrame({"call_idx": adj_call_idx, "adj": adj_cos})

            adj_min_call = adj_df.groupby("call_idx", sort=False)["adj"].min().reindex(np.arange(N), fill_value=0.0).to_numpy(np.float32)
            adj_std_call = adj_df.groupby("call_idx", sort=False)["adj"].std().reindex(np.arange(N), fill_value=0.0).fillna(0.0).to_numpy(np.float32)
            adj_last2_call = (
                adj_df.groupby("call_idx", sort=False).tail(2)
                .groupby("call_idx", sort=False)["adj"].mean()
                .reindex(np.arange(N), fill_value=0.0).to_numpy(np.float32)
            )

            adj_df["is_unique_next"] = (adj_df["adj"] < float(step_unique_thr)).astype(np.float32)
            unique_next_sum = adj_df.groupby("call_idx", sort=False)["is_unique_next"].sum().reindex(np.arange(N), fill_value=0.0).to_numpy(np.float32)
            adj_counts = adj_df.groupby("call_idx", sort=False)["adj"].size().reindex(np.arange(N), fill_value=0).to_numpy(np.int64)
            step_unique_ratio_call = (1.0 + unique_next_sum) / (1.0 + adj_counts.astype(np.float32) + 1e-9)

            chunk_to_cent = (E_chunks * call_emb_t[call_idx_t]).sum(dim=1).detach().cpu().numpy().astype(np.float32, copy=False)
            call_idx_chunks_np = call_idx_t.detach().cpu().numpy().astype(np.int64, copy=False)
            cent_df = pd.DataFrame({"call_idx": call_idx_chunks_np, "r": chunk_to_cent})
            cent_mean_call = cent_df.groupby("call_idx", sort=False)["r"].mean().reindex(np.arange(N), fill_value=0.0).to_numpy(np.float32)
            cent_min_call = cent_df.groupby("call_idx", sort=False)["r"].min().reindex(np.arange(N), fill_value=0.0).to_numpy(np.float32)

            pair_idx_np = d_top["pair_idx"].to_numpy(np.int64)
                        # Long-range repetition: does step j repeat any sufficiently earlier step?
            prev_repeat_frac_call = np.zeros((N,), dtype=np.float32)

            if "step_prev_repeat_frac" in requested:
                gap = max(int(step_prev_repeat_gap), 1)
                thr = float(step_prev_repeat_thr)

                # build per-call chunk embeddings
                # call_idx_chunks_np: for each chunk row in E_chunks, which call it belongs to
                # E_chunks: [Nc, D]
                call_to_rows = {}
                for row_idx, cidx in enumerate(call_idx_chunks_np):
                    call_to_rows.setdefault(int(cidx), []).append(row_idx)

                for cidx in range(N):
                    rows_idx = call_to_rows.get(int(cidx), [])
                    m = len(rows_idx)

                    # need at least one pair with index gap
                    if m <= gap:
                        prev_repeat_frac_call[cidx] = 0.0
                        continue

                    Xc = E_chunks[rows_idx].detach().cpu().numpy().astype(np.float32, copy=False)  # [m, D]
                    Sfull = Xc @ Xc.T  # cosine matrix because embeddings normalized

                    flags = []
                    for j in range(gap, m):
                        # previous steps up to j-gap
                        prev_max = float(Sfull[j, : (j - gap + 1)].max())
                        flags.append(1.0 if prev_max >= thr else 0.0)

                    prev_repeat_frac_call[cidx] = float(np.mean(flags)) if flags else 0.0

            call_feat_df = pd.DataFrame({
                "pair_idx": pair_idx_np,
                "adj_min": adj_min_call,
                "adj_std": adj_std_call,
                "adj_last2": adj_last2_call,
                "uniq_ratio": step_unique_ratio_call,
                "prev_repeat_frac": prev_repeat_frac_call,
                "cent_mean": cent_mean_call,
                "cent_min": cent_min_call,
            })

            if "step_adj_cos_min" in requested:
                out["step_adj_cos_min"] = (
                    call_feat_df.groupby("pair_idx", sort=False)["adj_min"]
                    .mean().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
                )

            if "step_adj_cos_std" in requested:
                out["step_adj_cos_std"] = (
                    call_feat_df.groupby("pair_idx", sort=False)["adj_std"]
                    .mean().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
                )

            if "step_adj_cos_last2" in requested:
                out["step_adj_cos_last2"] = (
                    call_feat_df.groupby("pair_idx", sort=False)["adj_last2"]
                    .mean().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
                )

            if "step_unique_step_ratio" in requested:
                out["step_unique_step_ratio"] = (
                    call_feat_df.groupby("pair_idx", sort=False)["uniq_ratio"]
                    .mean().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
                )

            if "step_prev_repeat_frac" in requested:
                out["step_prev_repeat_frac"] = (
                    call_feat_df.groupby("pair_idx", sort=False)["prev_repeat_frac"]
                    .mean().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
                )

            if "step_to_chain_centroid_mean" in requested:
                out["step_to_chain_centroid_mean"] = (
                    call_feat_df.groupby("pair_idx", sort=False)["cent_mean"]
                    .mean().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
                )

            if "step_to_chain_centroid_min" in requested:
                out["step_to_chain_centroid_min"] = (
                    call_feat_df.groupby("pair_idx", sort=False)["cent_min"]
                    .mean().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
                )

        # Q×step semantics
        need_qstep = any(f in requested for f in [
            "last_step_to_q_cos_mean", "last_step_to_q_cos_min",
            "q_step_cos_min_call_mean", "q_step_cos_max_call_mean", "q_step_cos_range_call_mean",
            "q_step_cos_last2_mean", "q_step_cos_last2_min",
        ])
        if need_qstep:
            q_for_call = Q_id[id_idx_call_t]  # [N,D]
            last_to_q = (E_last * q_for_call).sum(dim=1).detach().cpu().numpy()
            pair_idx_np = d_top["pair_idx"].to_numpy(np.int64)
            tmp_lq = pd.DataFrame({"pair_idx": pair_idx_np, "lq": last_to_q})
            if "last_step_to_q_cos_mean" in requested:
                out["last_step_to_q_cos_mean"] = tmp_lq.groupby("pair_idx", sort=False)["lq"].mean().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
            if "last_step_to_q_cos_min" in requested:
                out["last_step_to_q_cos_min"] = tmp_lq.groupby("pair_idx", sort=False)["lq"].min().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)

            # chunk->call->id->q mapping
            q_for_chunk = Q_id[id_idx_call_t[call_idx_t]]
            step_to_q = (E_chunks * q_for_chunk).sum(dim=1).detach().cpu().numpy()
            call_idx_chunks_np = call_idx_t.detach().cpu().numpy().astype(np.int64, copy=False)
            step_df = pd.DataFrame({"call_idx": call_idx_chunks_np, "sq": step_to_q})

            sq_min_call = step_df.groupby("call_idx", sort=False)["sq"].min().reindex(np.arange(N), fill_value=0.0).to_numpy(np.float32)
            sq_max_call = step_df.groupby("call_idx", sort=False)["sq"].max().reindex(np.arange(N), fill_value=0.0).to_numpy(np.float32)
            sq_range_call = sq_max_call - sq_min_call
            sq_last2_call = (
                step_df.groupby("call_idx", sort=False).tail(2)
                .groupby("call_idx", sort=False)["sq"].mean()
                .reindex(np.arange(N), fill_value=0.0).to_numpy(np.float32)
            )

            call_qstep_df = pd.DataFrame({
                "pair_idx": pair_idx_np,
                "sq_min": sq_min_call,
                "sq_max": sq_max_call,
                "sq_range": sq_range_call,
                "sq_last2": sq_last2_call,
            })

            if "q_step_cos_min_call_mean" in requested:
                out["q_step_cos_min_call_mean"] = call_qstep_df.groupby("pair_idx", sort=False)["sq_min"].mean().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
            if "q_step_cos_max_call_mean" in requested:
                out["q_step_cos_max_call_mean"] = call_qstep_df.groupby("pair_idx", sort=False)["sq_max"].mean().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
            if "q_step_cos_range_call_mean" in requested:
                out["q_step_cos_range_call_mean"] = call_qstep_df.groupby("pair_idx", sort=False)["sq_range"].mean().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
            if "q_step_cos_last2_mean" in requested:
                out["q_step_cos_last2_mean"] = call_qstep_df.groupby("pair_idx", sort=False)["sq_last2"].mean().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)
            if "q_step_cos_last2_min" in requested:
                out["q_step_cos_last2_min"] = call_qstep_df.groupby("pair_idx", sort=False)["sq_last2"].min().reindex(np.arange(M), fill_value=0.0).to_numpy(np.float32)

    # Safety fills + keep only requested
    for c in list(out.columns):
        if c in (id_col, answer_col):
            continue
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    keep = [id_col, answer_col] + [c for c in (requested_features or list(requested)) if c in out.columns]
    return out[keep].copy()

# ============================================================
# Prefix-convergence MiniLM feature builder (separate, cached)
# ============================================================

def _split_rationale_into_steps(text: str) -> List[str]:
    text = "" if pd.isna(text) else str(text).strip()
    if not text:
        return []

    parts = re.split(r'(?=Step\s*\d+\s*:)', text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        return parts

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 2:
        return lines

    sents = re.split(r'(?<=[\.\!\?])\s+', text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents if sents else [text]


def _build_step_prefixes(steps: List[str]) -> List[str]:
    prefixes = []
    cur = []
    for st in steps:
        cur.append(st)
        prefixes.append("\n".join(cur))
    return prefixes


def _build_word_prefixes(text: str, stride: int = 10, min_words: int = 5) -> Tuple[List[str], List[int]]:
    words = str(text).split()
    if not words:
        return [""], [0]

    idxs = list(range(min_words, len(words) + 1, stride))
    if not idxs:
        idxs = [len(words)]
    elif idxs[-1] != len(words):
        idxs.append(len(words))

    prefixes = [" ".join(words[:i]) for i in idxs]
    return prefixes, idxs


def _make_prefix_df_for_one_id(
    df_one_id: pd.DataFrame,
    id_col: str,
    answer_col: str,
    rationale_col: str,
    hit_col: str,
    mode: str = "step",
    word_stride: int = 10,
    min_words: int = 5,
) -> pd.DataFrame:
    d = df_one_id.copy().reset_index(drop=False).rename(columns={"index": "__trace_id__"})
    rows = []

    for _, row in d.iterrows():
        qid = row[id_col]
        ans = row[answer_col]
        rat = "" if pd.isna(row[rationale_col]) else str(row[rationale_col])
        hit = float(row[hit_col])
        trace_id = int(row["__trace_id__"])

        if mode == "step":
            steps = _split_rationale_into_steps(rat)
            if not steps:
                steps = [""]
            prefixes = _build_step_prefixes(steps)
            total = len(prefixes)

            for j, pref in enumerate(prefixes, start=1):
                rows.append({
                    id_col: qid,
                    "trace_id": trace_id,
                    answer_col: ans,
                    hit_col: hit,
                    "prefix_idx": j,
                    "n_prefixes": total,
                    "prefix_frac": j / max(total, 1),
                    "prefix_text": pref,
                })

        elif mode == "word":
            prefixes, idxs = _build_word_prefixes(rat, stride=word_stride, min_words=min_words)
            total_words = max(len(rat.split()), 1)

            for j, (pref, widx) in enumerate(zip(prefixes, idxs), start=1):
                rows.append({
                    id_col: qid,
                    "trace_id": trace_id,
                    answer_col: ans,
                    hit_col: hit,
                    "prefix_idx": j,
                    "n_prefixes": len(prefixes),
                    "prefix_frac": widx / total_words,
                    "prefix_text": pref,
                })
        else:
            raise ValueError("mode must be 'step' or 'word'")

    return pd.DataFrame(rows)


def _encode_with_cache_sentence_transformer(
    model,
    texts: List[str],
    kind: str,
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
    device: str,
    embed_cache: Dict,
) -> torch.Tensor:
    texts = ["" if t is None else str(t) for t in texts]
    if len(texts) == 0:
        return torch.empty((0, 0), dtype=torch.float32, device=device)

    gpu_cache = _get_ephemeral_gpu_cache(embed_cache, device)
    keys = [(kind, model_name, bool(normalize_embeddings), t) for t in texts]

    out_chunks: List[Optional[torch.Tensor]] = [None] * len(texts)
    miss_idx: List[int] = []
    miss_texts: List[str] = []

    for i, k in enumerate(keys):
        g = gpu_cache.get(k)
        if g is not None:
            out_chunks[i] = g
            continue

        cpu_val = embed_cache.get(k)
        if cpu_val is not None:
            t = torch.from_numpy(cpu_val).to(device=device, dtype=torch.float32, non_blocking=True)
            gpu_cache[k] = t
            out_chunks[i] = t
        else:
            miss_idx.append(i)
            miss_texts.append(texts[i])

    if miss_idx:
        miss_emb = model.encode(
            miss_texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )
        if miss_emb.dtype != torch.float32:
            miss_emb = miss_emb.float()

        miss_emb_cpu = miss_emb.detach().cpu().numpy().astype(np.float32, copy=False)
        for j, i in enumerate(miss_idx):
            k = keys[i]
            embed_cache[k] = miss_emb_cpu[j]
            gpu_cache[k] = miss_emb[j]
            out_chunks[i] = miss_emb[j]

    return torch.stack(out_chunks, dim=0)


def _summarize_prefix_convergence_one_id(
    prefix_df_one_id: pd.DataFrame,
    answer_col: str,
    depth_col: str,
    sim_threshold: float,
    depth_tol: float,
    early_frac: float,
    late_frac: float,
    rel_support_thr: float = 1.0 / 3.0,
    embeddings: Optional[torch.Tensor] = None,
) -> pd.DataFrame:
    rows = []

    for _, g_idx in prefix_df_one_id.groupby(answer_col, sort=False).groups.items():
        idx = np.asarray(g_idx, dtype=np.int64)
        g = prefix_df_one_id.iloc[idx].copy().reset_index(drop=True)
        ans = g[answer_col].iloc[0]

        trace_ids = list(pd.unique(g["trace_id"]))
        n_traces = len(trace_ids)
        n_prefix_points = len(g)

        if n_traces <= 1 or n_prefix_points == 0:
            rows.append({
                answer_col: ans,
                "prefix_n_traces": n_traces,
                "prefix_n_prefix_points": n_prefix_points,
                "prefix_converge_frac": 0.0,
                "prefix_converge_depth_mean": 0.0,
                "prefix_converge_depth_min": 0.0,
                "shared_checkpoint_count": 0.0,
                "shared_checkpoint_frac": 0.0,
                "shared_checkpoint_per_trace": 0.0,
                "max_checkpoint_coverage": 0.0,
                "mean_checkpoint_coverage": 0.0,
                "early_checkpoint_coverage": 0.0,
                "late_checkpoint_coverage": 0.0,
                "matched_trace_pair_frac": 0.0,
                "mean_prefixes_per_trace": float(n_prefix_points / max(n_traces, 1)),
            })
            continue

        if embeddings is None:
            E = torch.from_numpy(np.stack(g["embedding"].to_list(), axis=0)).float()
        else:
            E = embeddings[torch.as_tensor(idx, dtype=torch.long, device=embeddings.device)]
            if E.dtype != torch.float32:
                E = E.float()

        dev = E.device
        depth_t = torch.as_tensor(g[depth_col].to_numpy(dtype=np.float32, copy=False), device=dev)
        trace_codes_np, trace_uniques = pd.factorize(g["trace_id"], sort=False)
        trace_codes_t = torch.as_tensor(trace_codes_np, dtype=torch.long, device=dev)
        n_trace_pairs = max(n_traces - 1, 1)

        # keep cosine-similarity meaning identical to sklearn cosine_similarity
        E_norm = torch.nn.functional.normalize(E, p=2.0, dim=1, eps=1e-12)
        S = E_norm @ E_norm.T

        not_same_trace = trace_codes_t[:, None] != trace_codes_t[None, :]
        similar_depth = (depth_t[:, None] - depth_t[None, :]).abs() <= float(depth_tol)
        similar_semantics = S >= float(sim_threshold)
        checkpoint_adj = not_same_trace & similar_depth & similar_semantics
        
        # -------------------------------------------------
        # NEW: coverage by DISTINCT matched traces
        # so one other trace counts at most once per prefix
        # -------------------------------------------------
        support_count_t = torch.zeros((n_prefix_points,), dtype=torch.float32, device=dev)
        
        for i in range(n_prefix_points):
            matched_js = torch.where(checkpoint_adj[i])[0]
            if matched_js.numel() == 0:
                support_count_t[i] = 0.0
            else:
                matched_trace_codes = torch.unique(trace_codes_t[matched_js])
                support_count_t[i] = float(matched_trace_codes.numel())
        
        coverage_frac_t = support_count_t / float(n_trace_pairs)
        
        # weak checkpoint: matched by at least one other trace
        is_shared_t = coverage_frac_t > 0.0
        
        # strong checkpoint: matched by at least rel_support_thr of other traces
        is_shared_rel_t = coverage_frac_t >= float(rel_support_thr)
        
        # keep the original feature names, but now they use the strong definition
        shared_checkpoint_count = float(is_shared_rel_t.sum().item())
        shared_checkpoint_frac = shared_checkpoint_count / max(n_prefix_points, 1)
        shared_checkpoint_per_trace = shared_checkpoint_count / max(n_traces, 1)
        
        max_checkpoint_coverage = float(coverage_frac_t.max().item()) if n_prefix_points > 0 else 0.0
        mean_checkpoint_coverage = float(coverage_frac_t.mean().item()) if n_prefix_points > 0 else 0.0
        
        early_mask_t = depth_t <= float(early_frac)
        late_mask_t = depth_t >= float(late_frac)
        
        # keep early/late coverage as mean relative support
        early_checkpoint_coverage = (
            float(coverage_frac_t[early_mask_t].mean().item())
            if bool(early_mask_t.any().item()) else 0.0
        )
        late_checkpoint_coverage = (
            float(coverage_frac_t[late_mask_t].mean().item())
            if bool(late_mask_t.any().item()) else 0.0
        )

        trace_pair_hit = torch.zeros((n_traces, n_traces), dtype=torch.bool, device=dev)
        ii, jj = torch.where(checkpoint_adj)
        if ii.numel() > 0:
            ti = trace_codes_t[ii]
            tj = trace_codes_t[jj]
            trace_pair_hit[ti, tj] = True
            trace_pair_hit[tj, ti] = True
        matched_pairs = int(torch.triu(trace_pair_hit, diagonal=1).sum().item())
        denom_pairs = max(n_traces * (n_traces - 1) / 2.0, 1.0)
        matched_trace_pair_frac = float(matched_pairs / denom_pairs)

        # preserve original "first shared checkpoint depth per trace" meaning
        g["checkpoint_coverage"] = coverage_frac_t.detach().cpu().numpy().astype(np.float32, copy=False)
        g["is_shared_checkpoint"] = is_shared_rel_t.detach().cpu().numpy()

        first_shared_depths = []
        for _, gt in g.groupby("trace_id", sort=False):
            gt = gt.sort_values(depth_col, kind="mergesort")
            mask = gt["is_shared_checkpoint"].to_numpy()
            if mask.any():
                first_shared_depths.append(float(gt.loc[mask, depth_col].iloc[0]))

        if len(first_shared_depths) > 0:
            prefix_converge_frac = float(len(first_shared_depths) / max(n_traces, 1))
            prefix_converge_depth_mean = float(np.mean(first_shared_depths))
            prefix_converge_depth_min = float(np.min(first_shared_depths))
        else:
            prefix_converge_frac = 0.0
            prefix_converge_depth_mean = 0.0
            prefix_converge_depth_min = 0.0

        rows.append({
            answer_col: ans,
            "prefix_n_traces": n_traces,
            "prefix_n_prefix_points": n_prefix_points,
            "prefix_converge_frac": prefix_converge_frac,
            "prefix_converge_depth_mean": prefix_converge_depth_mean,
            "prefix_converge_depth_min": prefix_converge_depth_min,
            "shared_checkpoint_count": shared_checkpoint_count,
            "shared_checkpoint_frac": shared_checkpoint_frac,
            "shared_checkpoint_per_trace": shared_checkpoint_per_trace,
            "max_checkpoint_coverage": max_checkpoint_coverage,
            "mean_checkpoint_coverage": mean_checkpoint_coverage,
            "early_checkpoint_coverage": early_checkpoint_coverage,
            "late_checkpoint_coverage": late_checkpoint_coverage,
            "matched_trace_pair_frac": matched_trace_pair_frac,
            "mean_prefixes_per_trace": float(n_prefix_points / max(n_traces, 1)),
        })

    return pd.DataFrame(rows)

def add_id_answer_prefix_convergence_features_minilm(
    df: pd.DataFrame,
    top_n: int,
    id_col: str = "id",
    answer_col: str = "final_answer_new",
    rationale_col: str = "rationale",
    hit_col: str = "row_hit",
    call_col: Optional[str] = None,
    sort_cols: Optional[Sequence[str]] = None,
    ascending: bool = True,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
    device: str = "cuda",
    normalize_embeddings: bool = True,
    embed_cache: Optional[Dict] = None,
    mode: str = "step",
    word_stride: int = 10,
    min_words: int = 5,
    sim_threshold: float = 0.8,
    depth_tol: float = 0.20,
    early_frac: float = 0.33,
    late_frac: float = 0.66,
    rel_support_thr: float = 1.0 / 3.0,
    requested_features: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Separate feature builder for prefix-convergence features.
    Returns one row per (id, answer).
    Uses its own cache (distinct from add_id_answer_embedding_features_minilm).
    """
    if requested_features is None:
        requested = set(PREFIX_CONVERGENCE_FEATURES_MINILM_ALL)
    else:
        requested = set(requested_features)

    need = [id_col, answer_col, rationale_col, hit_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if embed_cache is None:
        embed_cache = {}

    d = df.copy()
    if sort_cols is not None:
        d = d.sort_values(
            [id_col] + list(sort_cols),
            ascending=[True] + [ascending] * len(sort_cols),
            kind="mergesort",
        )
    else:
        d = d.sort_values([id_col], kind="mergesort")

    if call_col is None:
        d["_call_idx"] = d.groupby(id_col, sort=False).cumcount() + 1
        t_col = "_call_idx"
    else:
        d[call_col] = pd.to_numeric(d[call_col], errors="coerce")
        d = d.dropna(subset=[call_col])
        t_col = call_col

    d = d.sort_values([id_col, t_col], kind="mergesort")
    d_top = d.groupby(id_col, sort=False, group_keys=False).head(top_n).copy()

    model = SentenceTransformer(model_name, device=device)

    parts = []
    for qid, gq in d_top.groupby(id_col, sort=False):
        prefix_df = _make_prefix_df_for_one_id(
            gq,
            id_col=id_col,
            answer_col=answer_col,
            rationale_col=rationale_col,
            hit_col=hit_col,
            mode=mode,
            word_stride=word_stride,
            min_words=min_words,
        )

        if len(prefix_df) == 0:
            continue

        E = _encode_with_cache_sentence_transformer(
            model=model,
            texts=prefix_df["prefix_text"].fillna("").astype(str).tolist(),
            kind=f"prefix::{mode}",
            model_name=model_name,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            device=device,
            embed_cache=embed_cache,
        )

        feat_q = _summarize_prefix_convergence_one_id(
            prefix_df_one_id=prefix_df,
            answer_col=answer_col,
            depth_col="prefix_frac",
            sim_threshold=sim_threshold,
            depth_tol=depth_tol,
            early_frac=early_frac,
            late_frac=late_frac,
            rel_support_thr=rel_support_thr,
            embeddings=E,
        )
        feat_q[id_col] = qid
        parts.append(feat_q)

    if parts:
        out = pd.concat(parts, ignore_index=True)
    else:
        out = d_top[[id_col, answer_col]].drop_duplicates().copy()
        for c in PREFIX_CONVERGENCE_FEATURES_MINILM_ALL:
            out[c] = 0.0

    keep = [id_col, answer_col] + [c for c in (requested_features or list(requested)) if c in out.columns]
    for c in keep:
        if c in (id_col, answer_col):
            continue
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out[keep].copy()



# ============================================================
# Feature switchboard config + build_features_for_topn
# ============================================================

@dataclass
class FeatureConfig:
    handcrafted_cols: List[str]
    use_embeddings: bool = True
    embedding_cols: Optional[List[str]] = None  # if None, use EMBED_FEATURES_BASE

    # NEW: prefix-convergence features
    use_prefix_embeddings: bool = False
    prefix_embedding_cols: Optional[List[str]] = None  # if None, use []
    prefix_mode: str = "step"          # "step" or "word"
    prefix_word_stride: int = 10
    prefix_min_words: int = 5
    prefix_sim_threshold: float = 0.8
    prefix_depth_tol: float = 0.20
    prefix_early_frac: float = 0.33
    prefix_late_frac: float = 0.66
    prefix_rel_support_thr: float = 1.0 / 3.0
    adaptive_prefix_thresholds: bool = False

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 512
    device: str = "cuda"
    normalize_embeddings: bool = True
    answer_centroid_weighted_by_calls: bool = True

    def resolved_embedding_cols(self) -> List[str]:
        if not self.use_embeddings:
            return []
        if self.embedding_cols is None:
            return list(EMBED_FEATURES_BASE)
        return list(self.embedding_cols)

    def resolved_prefix_embedding_cols(self) -> List[str]:
        if not self.use_prefix_embeddings:
            return []
        if self.prefix_embedding_cols is None:
            return []
        return list(self.prefix_embedding_cols)


def _compute_step_to_chain_centroid_min_multibudget(
    df_trace: pd.DataFrame,
    budgets: Sequence[int],
    feature_cfg: FeatureConfig,
    id_col: str = "id",
    answer_col: str = "final_answer_new",
    rationale_col: str = "rationale",
    call_col: Optional[str] = None,
    embed_cache: Optional[Dict] = None,
) -> pd.DataFrame:
    budgets = sorted(set(int(b) for b in budgets))
    if not budgets:
        return pd.DataFrame(columns=[id_col, answer_col, "budget_n", "step_to_chain_centroid_min"])

    max_budget = max(budgets)
    d = df_trace.copy()
    if call_col is None:
        d["_call_idx"] = d.groupby(id_col, sort=False).cumcount() + 1
        t_col = "_call_idx"
    else:
        d[call_col] = pd.to_numeric(d[call_col], errors="coerce")
        d = d.dropna(subset=[call_col])
        t_col = call_col
    d = d.sort_values([id_col, t_col], kind="mergesort")
    d_top = d.groupby(id_col, sort=False, group_keys=False).head(max_budget).copy()
    if d_top.empty:
        return pd.DataFrame(columns=[id_col, answer_col, "budget_n", "step_to_chain_centroid_min"])

    if embed_cache is None:
        embed_cache = {}

    model = SentenceTransformer(feature_cfg.model_name, device=feature_cfg.device)

    @torch.no_grad()
    def _encode_with_cache(texts: List[str], kind: str) -> torch.Tensor:
        keys = [(kind, t) for t in texts]
        miss_idx = [i for i, k in enumerate(keys) if k not in embed_cache]
        if miss_idx:
            miss_texts = [texts[i] for i in miss_idx]
            miss_emb = model.encode(
                miss_texts,
                batch_size=feature_cfg.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=feature_cfg.normalize_embeddings,
                show_progress_bar=False,
            )
            if miss_emb.dtype != torch.float32:
                miss_emb = miss_emb.float()
            miss_np = miss_emb.detach().cpu().numpy().astype(np.float32, copy=False)
            for j, i in enumerate(miss_idx):
                embed_cache[keys[i]] = miss_np[j]
        if not keys:
            return torch.empty((0, 384), dtype=torch.float32, device=feature_cfg.device)
        D = int(embed_cache[keys[0]].shape[0])
        out_np = np.empty((len(texts), D), dtype=np.float32)
        for i, k in enumerate(keys):
            out_np[i] = embed_cache[k]
        return torch.from_numpy(out_np).to(feature_cfg.device)

    _dot_split = re.compile(r"\.(?=\s|$)")

    def split_by_dot(text: str) -> List[str]:
        t = (text or "").strip()
        if not t:
            return [""]
        t = re.sub(r"\s+", " ", t)
        parts = _dot_split.split(t)
        chunks: List[str] = []
        for p in parts:
            s = p.strip()
            if not s:
                continue
            if not s.endswith("."):
                s += "."
            if len(s) < 3:
                continue
            chunks.append(s)
            if len(chunks) >= 64:
                break
        return chunks or [t[:4000]]

    d_top["_cot_text"] = d_top[rationale_col].fillna("").astype(str)
    uniq = (
        d_top.groupby([id_col, answer_col], as_index=False, dropna=False)
        .size()
        .rename(columns={"size": "ans_calls"})
    )
    uniq["pair_idx"] = np.arange(len(uniq), dtype=np.int64)
    d_top = d_top.merge(uniq[[id_col, answer_col, "pair_idx"]], on=[id_col, answer_col], how="left", sort=False)

    chunk_texts: List[str] = []
    chunk_call_idx: List[int] = []
    for i, txt in enumerate(d_top["_cot_text"].tolist()):
        for ch in split_by_dot(txt):
            chunk_texts.append(ch)
            chunk_call_idx.append(i)

    if not chunk_texts:
        rows = []
        for b in budgets:
            tmp = uniq[[id_col, answer_col]].copy()
            tmp["budget_n"] = int(b)
            tmp["step_to_chain_centroid_min"] = 0.0
            rows.append(tmp)
        return pd.concat(rows, ignore_index=True)

    E_chunks = _encode_with_cache(chunk_texts, kind="cot_chunk")
    if E_chunks.dtype != torch.float32:
        E_chunks = E_chunks.float()
    device = E_chunks.device
    call_idx_t = torch.as_tensor(np.asarray(chunk_call_idx, dtype=np.int64), device=device)
    N = len(d_top)
    D = E_chunks.shape[1]
    counts = torch.bincount(call_idx_t, minlength=N).to(device=device, dtype=torch.float32)
    sums = torch.zeros((N, D), device=device, dtype=torch.float32)
    sums.index_add_(0, call_idx_t, E_chunks)
    call_emb_t = sums / counts.clamp_min(1.0).unsqueeze(1)
    chunk_to_cent = (E_chunks * call_emb_t[call_idx_t]).sum(dim=1)

    cent_min_call = torch.full((N,), float('inf'), device=device, dtype=torch.float32)
    cent_min_call.scatter_reduce_(0, call_idx_t, chunk_to_cent, reduce='amin', include_self=True)
    cent_min_call = torch.where(torch.isfinite(cent_min_call), cent_min_call, torch.zeros_like(cent_min_call))
    cent_min_call_np = cent_min_call.detach().cpu().numpy().astype(np.float32, copy=False)

    pair_idx = pd.to_numeric(d_top["pair_idx"], errors="coerce").fillna(-1).to_numpy(np.int64, copy=False)
    call_ord = pd.to_numeric(d_top[t_col], errors="coerce").fillna(0).to_numpy(np.int64, copy=False)
    order = np.lexsort((call_ord, pair_idx))
    pair_idx_s = pair_idx[order]
    call_ord_s = call_ord[order]
    cent_s = cent_min_call_np[order]

    uniq_pair_idx, start_idx, counts_pair = np.unique(pair_idx_s, return_index=True, return_counts=True)
    end_idx = start_idx + counts_pair
    pair_id = uniq.loc[uniq_pair_idx, id_col].to_numpy(copy=False)
    pair_ans = uniq.loc[uniq_pair_idx, answer_col].to_numpy(copy=False)
    budgets_np = np.asarray(budgets, dtype=np.int64)

    out_id: List[Any] = []
    out_ans: List[Any] = []
    out_budget: List[int] = []
    out_val: List[float] = []

    for row_i, s, e in zip(range(len(uniq_pair_idx)), start_idx, end_idx):
        ord_g = call_ord_s[s:e]
        val_g = cent_s[s:e].astype(np.float64, copy=False)
        csum = np.cumsum(val_g, dtype=np.float64)
        k_arr = np.searchsorted(ord_g, budgets_np, side='right')
        vals = np.zeros(len(budgets_np), dtype=np.float32)
        valid = k_arr > 0
        if np.any(valid):
            vals[valid] = (csum[k_arr[valid] - 1] / k_arr[valid]).astype(np.float32, copy=False)
        out_id.extend([pair_id[row_i]] * len(budgets_np))
        out_ans.extend([pair_ans[row_i]] * len(budgets_np))
        out_budget.extend(budgets_np.tolist())
        out_val.extend(vals.tolist())

    return pd.DataFrame({
        id_col: out_id,
        answer_col: out_ans,
        "budget_n": out_budget,
        "step_to_chain_centroid_min": out_val,
    })


def _compute_shared_checkpoint_count_multibudget(
    df_trace: pd.DataFrame,
    budgets: Sequence[int],
    feature_cfg: FeatureConfig,
    id_col: str = "id",
    answer_col: str = "final_answer_new",
    rationale_col: str = "rationale",
    hit_col: str = "row_hit",
    call_col: Optional[str] = None,
    prefix_embed_cache: Optional[Dict] = None,
) -> pd.DataFrame:
    budgets = sorted(set(int(b) for b in budgets))
    if not budgets:
        return pd.DataFrame(columns=[id_col, answer_col, "budget_n", "shared_checkpoint_count"])
    max_budget = max(budgets)

    d = df_trace.copy()
    if call_col is None:
        d["_call_idx"] = d.groupby(id_col, sort=False).cumcount() + 1
        t_col = "_call_idx"
    else:
        d[call_col] = pd.to_numeric(d[call_col], errors="coerce")
        d = d.dropna(subset=[call_col])
        t_col = call_col
    d = d.sort_values([id_col, t_col], kind="mergesort")
    d_top = d.groupby(id_col, sort=False, group_keys=False).head(max_budget).copy()
    if d_top.empty:
        return pd.DataFrame(columns=[id_col, answer_col, "budget_n", "shared_checkpoint_count"])

    if prefix_embed_cache is None:
        prefix_embed_cache = {}
    model = SentenceTransformer(feature_cfg.model_name, device=feature_cfg.device)
    parts: List[pd.DataFrame] = []

    for qid, gq in d_top.groupby(id_col, sort=False):
        prefix_df = _make_prefix_df_for_one_id(
            gq,
            id_col=id_col,
            answer_col=answer_col,
            rationale_col=rationale_col,
            hit_col=hit_col,
            mode=feature_cfg.prefix_mode,
            word_stride=feature_cfg.prefix_word_stride,
            min_words=feature_cfg.prefix_min_words,
        )
        gq_local = gq.reset_index(drop=True).copy()
        gq_local["trace_id"] = np.arange(len(gq_local), dtype=np.int64)
        gq_local["call_idx"] = np.arange(1, len(gq_local) + 1, dtype=np.int64)
        prefix_df = prefix_df.merge(
            gq_local[["trace_id", "call_idx"]],
            on="trace_id",
            how="left",
            validate="many_to_one",
        )
        if len(prefix_df) == 0:
            continue

        E = _encode_with_cache_sentence_transformer(
            model=model,
            texts=prefix_df["prefix_text"].fillna("").astype(str).tolist(),
            kind=f"prefix::{feature_cfg.prefix_mode}",
            model_name=feature_cfg.model_name,
            batch_size=feature_cfg.batch_size,
            normalize_embeddings=feature_cfg.normalize_embeddings,
            device=feature_cfg.device,
            embed_cache=prefix_embed_cache,
        )
        if E.dtype != torch.float32:
            E = E.float()
        E = torch.nn.functional.normalize(E, p=2.0, dim=1, eps=1e-12)
        device = E.device
        S_full = E @ E.T
        depth_full = torch.as_tensor(
            prefix_df["prefix_frac"].to_numpy(dtype=np.float32, copy=False),
            dtype=torch.float32,
            device=device,
        )
        call_ord_full = pd.to_numeric(prefix_df["call_idx"], errors="coerce").fillna(0).to_numpy(np.int64, copy=False)

        answer_artifacts = []
        for ans, g_ans in prefix_df.groupby(answer_col, sort=False):
            idx_local_np = g_ans.index.to_numpy(dtype=np.int64, copy=False)
            idx_local_t = torch.as_tensor(idx_local_np, dtype=torch.long, device=device)
            call_ord_ans = call_ord_full[idx_local_np]
            depth_ans_full = depth_full.index_select(0, idx_local_t)
            S_ans_full = S_full.index_select(0, idx_local_t).index_select(1, idx_local_t)
            trace_ids_ans = g_ans["trace_id"].to_numpy(copy=False)
            answer_artifacts.append((ans, call_ord_ans, depth_ans_full, S_ans_full, trace_ids_ans))

        rows_for_q = []
        for b in budgets:
            sim_thr, rel_thr = _resolve_prefix_thresholds(feature_cfg, int(b))
            for ans, call_ord_ans, depth_ans_full, S_ans_full, trace_ids_ans in answer_artifacts:
                sel_idx_np = np.flatnonzero(call_ord_ans <= int(b))
                if sel_idx_np.size == 0:
                    rows_for_q.append((qid, ans, int(b), 0.0))
                    continue

                trace_ids_sel = trace_ids_ans[sel_idx_np]
                _, tc_ans_np = np.unique(trace_ids_sel, return_inverse=True)
                n_traces = int(tc_ans_np.max()) + 1 if tc_ans_np.size > 0 else 0
                if n_traces <= 1:
                    rows_for_q.append((qid, ans, int(b), 0.0))
                    continue

                sel_idx_t = torch.as_tensor(sel_idx_np, dtype=torch.long, device=device)
                S_ans = S_ans_full.index_select(0, sel_idx_t).index_select(1, sel_idx_t)
                depth_ans = depth_ans_full.index_select(0, sel_idx_t)
                tc_ans = torch.as_tensor(tc_ans_np, dtype=torch.long, device=device)

                n_other = max(n_traces - 1, 1)
                not_same_trace = tc_ans[:, None] != tc_ans[None, :]
                similar_depth = (depth_ans[:, None] - depth_ans[None, :]).abs() <= float(feature_cfg.prefix_depth_tol)
                similar_semantics = S_ans >= float(sim_thr)
                checkpoint_adj = not_same_trace & similar_depth & similar_semantics
                trace_one_hot = torch.nn.functional.one_hot(tc_ans, num_classes=n_traces).to(torch.float32)
                matched_trace_mask = (checkpoint_adj.to(torch.float32) @ trace_one_hot) > 0.0
                matched_trace_mask &= ~trace_one_hot.bool()
                support_count = matched_trace_mask.sum(dim=1).to(torch.float32)
                coverage = support_count / float(n_other)
                if float(rel_thr) <= 0.0:
                    is_shared_rel = coverage > 0.0
                else:
                    is_shared_rel = coverage >= float(rel_thr)
                rows_for_q.append((qid, ans, int(b), float(is_shared_rel.sum().item())))

        if rows_for_q:
            parts.append(pd.DataFrame(rows_for_q, columns=[id_col, answer_col, "budget_n", "shared_checkpoint_count"]))

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[id_col, answer_col, "budget_n", "shared_checkpoint_count"])


def build_features_for_budgets(
    df_trace: pd.DataFrame,
    budgets: Sequence[int],
    feature_cfg: FeatureConfig,
    id_col: str = "id",
    question_col: str = "question",
    answer_col: str = "final_answer_new",
    rationale_col: str = "rationale",
    hit_col: str = "row_hit",
    call_col: Optional[str] = None,
    caches: Optional[EmbeddingCaches] = None,
) -> pd.DataFrame:
    budgets = sorted(set(int(b) for b in budgets))
    if not budgets:
        raise ValueError("budgets must be non-empty")
    if caches is None:
        caches = EmbeddingCaches()

    parts: List[pd.DataFrame] = []
    emb_cols = feature_cfg.resolved_embedding_cols()
    prefix_cols = feature_cfg.resolved_prefix_embedding_cols()

    optimized_emb_cols = [c for c in emb_cols if c == "step_to_chain_centroid_min"]
    remaining_emb_cols = [c for c in emb_cols if c not in set(optimized_emb_cols)]

    optimized_prefix_cols = [c for c in prefix_cols if c == "shared_checkpoint_count"]
    remaining_prefix_cols = [c for c in prefix_cols if c not in set(optimized_prefix_cols)]

    opt_step_df = None
    if optimized_emb_cols:
        opt_step_df = _compute_step_to_chain_centroid_min_multibudget(
            df_trace=df_trace, budgets=budgets, feature_cfg=feature_cfg, id_col=id_col,
            answer_col=answer_col, rationale_col=rationale_col, call_col=call_col, embed_cache=caches.base_cache,
        )

    opt_prefix_df = None
    if optimized_prefix_cols:
        opt_prefix_df = _compute_shared_checkpoint_count_multibudget(
            df_trace=df_trace, budgets=budgets, feature_cfg=feature_cfg, id_col=id_col,
            answer_col=answer_col, rationale_col=rationale_col, hit_col=hit_col, call_col=call_col,
            prefix_embed_cache=caches.prefix_cache,
        )

    for b in budgets:
        feat_b = build_id_answer_features_selected(
            df_trace, top_n=int(b), feature_cols=feature_cfg.handcrafted_cols, id_col=id_col,
            answer_col=answer_col, hit_col=hit_col, rationale_col=rationale_col, call_col=call_col, sort_cols=None,
        )

        if remaining_emb_cols:
            emb_cfg = feature_cfg
            emb = add_id_answer_embedding_features_minilm(
                df_trace, top_n=int(b), id_col=id_col, question_col=question_col, answer_col=answer_col,
                rationale_col=rationale_col, call_col=call_col, model_name=feature_cfg.model_name,
                batch_size=feature_cfg.batch_size, device=feature_cfg.device,
                normalize_embeddings=feature_cfg.normalize_embeddings,
                answer_centroid_weighted_by_calls=feature_cfg.answer_centroid_weighted_by_calls,
                embed_cache=caches.base_cache, requested_features=remaining_emb_cols,
            )
            feat_b = feat_b.merge(emb, on=[id_col, answer_col], how="left")

        if opt_step_df is not None:
            feat_b = feat_b.merge(
                opt_step_df.loc[opt_step_df["budget_n"] == int(b), [id_col, answer_col, "step_to_chain_centroid_min"]],
                on=[id_col, answer_col], how="left"
            )

        if remaining_prefix_cols:
            sim_thr, rel_thr = _resolve_prefix_thresholds(feature_cfg, int(b))
            pfx = add_id_answer_prefix_convergence_features_minilm(
                df_trace, top_n=int(b), id_col=id_col, answer_col=answer_col, rationale_col=rationale_col,
                hit_col=hit_col, call_col=call_col, model_name=feature_cfg.model_name,
                batch_size=feature_cfg.batch_size, device=feature_cfg.device,
                normalize_embeddings=feature_cfg.normalize_embeddings, embed_cache=caches.prefix_cache,
                mode=feature_cfg.prefix_mode, word_stride=feature_cfg.prefix_word_stride,
                min_words=feature_cfg.prefix_min_words, sim_threshold=sim_thr, depth_tol=feature_cfg.prefix_depth_tol,
                early_frac=feature_cfg.prefix_early_frac, late_frac=feature_cfg.prefix_late_frac,
                rel_support_thr=rel_thr, requested_features=remaining_prefix_cols,
            )
            feat_b = feat_b.merge(pfx, on=[id_col, answer_col], how="left")

        if opt_prefix_df is not None:
            feat_b = feat_b.merge(
                opt_prefix_df.loc[opt_prefix_df["budget_n"] == int(b), [id_col, answer_col, "shared_checkpoint_count"]],
                on=[id_col, answer_col], how="left"
            )

        feat_b["budget_n"] = int(b)
        parts.append(feat_b)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    numeric_cols = [c for c in out.columns if c not in {id_col, answer_col}]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def build_features_for_topn(
    df_trace: pd.DataFrame,
    top_n: int,
    feature_cfg: FeatureConfig,
    id_col: str = "id",
    question_col: str = "question",
    answer_col: str = "final_answer_new",
    rationale_col: str = "rationale",
    hit_col: str = "row_hit",
    call_col: Optional[str] = None,
    embed_cache: Optional[Dict] = None,
    prefix_embed_cache: Optional[Dict] = None,
) -> pd.DataFrame:
    caches = EmbeddingCaches(base_cache=embed_cache, prefix_cache=prefix_embed_cache, runtime_cache={})
    out = build_features_for_budgets(
        df_trace=df_trace, budgets=[int(top_n)], feature_cfg=feature_cfg, id_col=id_col, question_col=question_col,
        answer_col=answer_col, rationale_col=rationale_col, hit_col=hit_col, call_col=call_col, caches=caches,
    )
    if "budget_n" in out.columns:
        out = out.drop(columns=["budget_n"])
    return out



# ============================================================
# Training / inference + bundle I/O
# ============================================================

@dataclass
class RerankerBundle:
    model: Any
    feature_cols: List[str]
    feature_cfg: FeatureConfig
    max_budget: int
    lgb_params: Dict[str, Any]

    def save(self, path: str | Path) -> None:
        import joblib
        payload = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "feature_cfg": asdict(self.feature_cfg),
            "max_budget": self.max_budget,
            "lgb_params": self.lgb_params,
        }
        joblib.dump(payload, str(path))

    @staticmethod
    def load(path: str | Path) -> "RerankerBundle":
        import joblib
        payload = joblib.load(str(path))
        fc = FeatureConfig(**payload["feature_cfg"])
        return RerankerBundle(
            model=payload["model"],
            feature_cols=list(payload["feature_cols"]),
            feature_cfg=fc,
            max_budget=int(payload["max_budget"]),
            lgb_params=dict(payload.get("lgb_params", {})),
        )


def _make_group_sizes(df_sorted: pd.DataFrame, qid_col: str) -> np.ndarray:
    return df_sorted.groupby(qid_col, sort=False).size().to_numpy(dtype=np.int32)


@dataclass
class RankerPreparedData:
    df_feat: pd.DataFrame
    feature_cols: List[str]
    max_budget: int
    qid_col: str
    orig_group_col: str
    target_col: str


def _extract_ranker_arrays(
    df_feat: pd.DataFrame,
    feature_cols: Sequence[str],
    qid_col: str,
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_feat = df_feat.sort_values([qid_col], kind="mergesort").reset_index(drop=True)
    X = df_feat[list(feature_cols)].to_numpy(dtype=float)
    y = df_feat[target_col].to_numpy(dtype=float)
    group = _make_group_sizes(df_feat, qid_col)
    return X, y, group


def _split_orig_ids(
    df_feat: pd.DataFrame,
    orig_group_col: str = "orig_id",
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[set, set]:
    orig_ids = df_feat[orig_group_col].drop_duplicates().to_numpy()
    if len(orig_ids) < 2:
        raise ValueError("Need at least 2 distinct orig_id values for train/validation split.")

    rng = np.random.default_rng(random_state)
    rng.shuffle(orig_ids)

    n_val = max(1, int(round(len(orig_ids) * val_size)))
    n_val = min(n_val, len(orig_ids) - 1)

    val_orig_ids = set(orig_ids[:n_val])
    train_orig_ids = set(orig_ids[n_val:])
    return train_orig_ids, val_orig_ids


def prepare_lgbm_reranker_multibudget_features(
    df_train_trace: pd.DataFrame,
    budgets: Sequence[int],
    feature_cfg: FeatureConfig,
    id_col: str = "id",
    question_col: str = "question",
    answer_col: str = "final_answer_new",
    rationale_col: str = "rationale",
    hit_col: str = "row_hit",
    call_col: Optional[str] = None,
    caches: Optional[EmbeddingCaches] = None,
) -> RankerPreparedData:
    budgets = sorted(set(int(b) for b in budgets))
    if not budgets:
        raise ValueError("budgets must be non-empty")
    max_budget = max(budgets)

    if caches is None:
        caches = EmbeddingCaches()

    df_feat = build_features_for_budgets(
        df_trace=df_train_trace,
        budgets=budgets,
        feature_cfg=feature_cfg,
        id_col=id_col,
        question_col=question_col,
        answer_col=answer_col,
        rationale_col=rationale_col,
        hit_col=hit_col,
        call_col=call_col,
        caches=caches,
    )
    df_feat["orig_id"] = df_feat[id_col]
    df_feat["budget_n_norm"] = df_feat["budget_n"] / float(max_budget)
    df_feat[id_col] = df_feat["orig_id"].astype(str) + "__" + df_feat["budget_n"].astype(str)

    emb_cols = feature_cfg.resolved_embedding_cols()
    prefix_cols = feature_cfg.resolved_prefix_embedding_cols()
    feature_cols = list(feature_cfg.handcrafted_cols) + emb_cols + prefix_cols + ["budget_n_norm"]

    for leak_col in ["hit_mean", hit_col, "row_hit"]:
        if leak_col in feature_cols:
            raise ValueError(f"Leakage: {leak_col} is in feature_cols.")

    df_feat = df_feat.sort_values([id_col], kind="mergesort").reset_index(drop=True)

    return RankerPreparedData(
        df_feat=df_feat,
        feature_cols=feature_cols,
        max_budget=max_budget,
        qid_col=id_col,
        orig_group_col="orig_id",
        target_col="hit_mean",
    )


def fit_lgbm_ranker_on_prepared_data(
    prepared: RankerPreparedData,
    feature_cfg: FeatureConfig,
    lgbm_ranker_ctor,
    lgbm_params: Dict[str, Any],
) -> Tuple[RerankerBundle, pd.DataFrame]:
    X, y, group = _extract_ranker_arrays(
        df_feat=prepared.df_feat,
        feature_cols=prepared.feature_cols,
        qid_col=prepared.qid_col,
        target_col=prepared.target_col,
    )

    model = lgbm_ranker_ctor(**lgbm_params)
    model.fit(X, y, group=group)

    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    imp = (
        pd.DataFrame({"feature": prepared.feature_cols, "gain": gain, "split": split})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    bundle = RerankerBundle(
        model=model,
        feature_cols=prepared.feature_cols,
        feature_cfg=feature_cfg,
        max_budget=prepared.max_budget,
        lgb_params=dict(lgbm_params),
    )
    return bundle, imp


def _dcg_at_k(rels: np.ndarray, k: int) -> float:
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    gains = (2.0 ** rels - 1.0) * discounts
    return float(np.sum(gains))


def _ndcg_at_k_per_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    k: int,
) -> float:
    vals: List[float] = []
    start = 0
    for g in group:
        end = start + int(g)
        yt = y_true[start:end]
        yp = y_pred[start:end]
        if len(yt) == 0:
            vals.append(0.0)
            start = end
            continue
        pred_order = np.argsort(-yp)
        true_order = np.argsort(-yt)
        dcg = _dcg_at_k(yt[pred_order], k)
        idcg = _dcg_at_k(yt[true_order], k)
        vals.append(0.0 if idcg <= 0.0 else dcg / idcg)
        start = end
    return float(np.mean(vals)) if vals else 0.0


def _expand_param_grid(param_grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    if not param_grid:
        return []
    keys = list(param_grid.keys())
    values = [list(param_grid[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def search_lgbm_ranker_params_on_validation(
    prepared: RankerPreparedData,
    lgbm_ranker_ctor,
    base_params: Dict[str, Any],
    param_grid: Dict[str, Sequence[Any]],
    val_size: float = 0.2,
    random_state: int = 42,
    ndcg_at: int = 10,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    df_feat = prepared.df_feat

    train_orig_ids, val_orig_ids = _split_orig_ids(
        df_feat,
        orig_group_col=prepared.orig_group_col,
        val_size=val_size,
        random_state=random_state,
    )

    df_tr = df_feat[df_feat[prepared.orig_group_col].isin(train_orig_ids)].copy()
    df_va = df_feat[df_feat[prepared.orig_group_col].isin(val_orig_ids)].copy()

    X_tr, y_tr, g_tr = _extract_ranker_arrays(df_tr, prepared.feature_cols, prepared.qid_col, prepared.target_col)
    X_va, y_va, g_va = _extract_ranker_arrays(df_va, prepared.feature_cols, prepared.qid_col, prepared.target_col)

    candidates = [dict()] + _expand_param_grid(param_grid)
    rows: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_params = dict(base_params)

    from tqdm.auto import tqdm

    for trial_idx, delta in enumerate(
        tqdm(candidates, desc="Hyperparameter search", unit="trial")):
        
        params = {**base_params, **delta}
        model = lgbm_ranker_ctor(**params)
        model.fit(X_tr, y_tr, group=g_tr)
        pred_va = model.predict(X_va)
        val_score = _ndcg_at_k_per_group(y_va, pred_va, g_va, k=ndcg_at)

        row: Dict[str, Any] = {"trial": trial_idx, "val_ndcg_at": ndcg_at, "val_score": float(val_score)}
        row.update(delta)
        rows.append(row)

        if val_score > best_score:
            best_score = float(val_score)
            best_params = dict(params)

    results_df = pd.DataFrame(rows).sort_values(["val_score", "trial"], ascending=[False, True]).reset_index(drop=True)
    return best_params, results_df


def train_lgbm_reranker_multibudget_with_hparam_search(
    df_train_trace: pd.DataFrame,
    budgets: Sequence[int],
    feature_cfg: FeatureConfig,
    lgbm_ranker_ctor,
    lgbm_params: Dict[str, Any],
    param_grid: Dict[str, Sequence[Any]],
    val_size: float = 0.2,
    random_state: int = 42,
    ndcg_at: int = 10,
    id_col: str = "id",
    question_col: str = "question",
    answer_col: str = "final_answer_new",
    rationale_col: str = "rationale",
    hit_col: str = "row_hit",
    call_col: Optional[str] = None,
    caches: Optional[EmbeddingCaches] = None,
) -> Tuple[RerankerBundle, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    prepared = prepare_lgbm_reranker_multibudget_features(
        df_train_trace=df_train_trace,
        budgets=budgets,
        feature_cfg=feature_cfg,
        id_col=id_col,
        question_col=question_col,
        answer_col=answer_col,
        rationale_col=rationale_col,
        hit_col=hit_col,
        call_col=call_col,
        caches=caches,
    )

    best_params, search_results = search_lgbm_ranker_params_on_validation(
        prepared=prepared,
        lgbm_ranker_ctor=lgbm_ranker_ctor,
        base_params=lgbm_params,
        param_grid=param_grid,
        val_size=val_size,
        random_state=random_state,
        ndcg_at=ndcg_at,
    )

    bundle, imp = fit_lgbm_ranker_on_prepared_data(
        prepared=prepared,
        feature_cfg=feature_cfg,
        lgbm_ranker_ctor=lgbm_ranker_ctor,
        lgbm_params=best_params,
    )

    return bundle, imp, search_results, best_params


def train_lgbm_reranker_multibudget(
    df_train_trace: pd.DataFrame,
    budgets: Sequence[int],
    feature_cfg: FeatureConfig,
    lgbm_ranker_ctor,
    lgbm_params: Dict[str, Any],
    id_col: str = "id",
    question_col: str = "question",
    answer_col: str = "final_answer_new",
    rationale_col: str = "rationale",
    hit_col: str = "row_hit",
    call_col: Optional[str] = None,
    caches: Optional[EmbeddingCaches] = None,
) -> Tuple[RerankerBundle, pd.DataFrame]:
    """Builds the multibudget training set and fits LGBMRanker."""
    import lightgbm as lgb  # local import

    prepared = prepare_lgbm_reranker_multibudget_features(
        df_train_trace=df_train_trace,
        budgets=budgets,
        feature_cfg=feature_cfg,
        id_col=id_col,
        question_col=question_col,
        answer_col=answer_col,
        rationale_col=rationale_col,
        hit_col=hit_col,
        call_col=call_col,
        caches=caches,
    )

    df_feat = prepared.df_feat
    feature_cols = prepared.feature_cols
    max_budget = prepared.max_budget

    df_feat = df_feat.sort_values([prepared.qid_col], kind="mergesort").reset_index(drop=True)
    X = df_feat[feature_cols].to_numpy(dtype=float)
    y = df_feat[prepared.target_col].to_numpy(dtype=float)
    group = _make_group_sizes(df_feat, prepared.qid_col)

    model = lgbm_ranker_ctor(**lgbm_params)
    model.fit(X, y, group=group)

    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    imp = (
        pd.DataFrame({"feature": feature_cols, "gain": gain, "split": split})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    bundle = RerankerBundle(
        model=model,
        feature_cols=feature_cols,
        feature_cfg=feature_cfg,
        max_budget=max_budget,
        lgb_params=lgbm_params,
    )
    return bundle, imp


def predict_topn(
    bundle: RerankerBundle,
    df_trace: pd.DataFrame,
    top_n: int,
    id_col: str = "id",
    question_col: str = "question",
    answer_col: str = "final_answer_new",
    rationale_col: str = "rationale",
    hit_col: str = "row_hit",
    call_col: Optional[str] = None,
    embed_cache: Optional[Dict] = None,
    prefix_embed_cache: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Returns one row per id (argmax answer) with score + hit_mean.
    """
    df_feat = build_features_for_topn(
        df_trace,
        top_n=top_n,
        feature_cfg=bundle.feature_cfg,
        id_col=id_col,
        question_col=question_col,
        answer_col=answer_col,
        rationale_col=rationale_col,
        hit_col=hit_col,
        call_col=call_col,
        embed_cache=embed_cache,
        prefix_embed_cache=prefix_embed_cache,
    )
    df_feat["budget_n_norm"] = float(top_n) / float(bundle.max_budget)
    X = df_feat[bundle.feature_cols].to_numpy(dtype=float)
    scores = bundle.model.predict(X)
    df_feat = df_feat.copy()
    df_feat["score"] = scores
    pred_idx = df_feat.groupby(id_col)["score"].idxmax()
    return df_feat.loc[pred_idx].reset_index(drop=True)


def evaluate_over_budgets(
    bundle: RerankerBundle,
    df_test_trace: pd.DataFrame,
    topn_values: Iterable[int],
    id_col: str = "id",
    question_col: str = "question",
    answer_col: str = "final_answer_new",
    rationale_col: str = "rationale",
    hit_col: str = "row_hit",
    call_col: Optional[str] = None,
    show_progress: bool = True,
    caches: Optional[EmbeddingCaches] = None,
) -> List[float]:
    """
    Returns list of mean(hit_mean) over predicted answers for each top_n.
    """
    if caches is None:
        caches = EmbeddingCaches()

    base_cache = caches.base_cache
    prefix_cache = caches.prefix_cache
    
    out: List[float] = []
    iterator = tqdm(list(topn_values), desc="Evaluating budgets", unit="top_n") if show_progress else topn_values

    for n in iterator:
        pred_rows = predict_topn(
        bundle=bundle,
        df_trace=df_test_trace,
        top_n=int(n),
        id_col=id_col,
        question_col=question_col,
        answer_col=answer_col,
        rationale_col=rationale_col,
        hit_col=hit_col,
        call_col=call_col,
        embed_cache=base_cache,
        prefix_embed_cache=prefix_cache,)
        
        score = float(pred_rows["hit_mean"].mean())
        out.append(score)
        if show_progress:
            iterator.set_postfix(top_n=int(n), mean_hit=score, running_mean=sum(out) / len(out))

    return out


def save_scores_list(scores: Sequence[float], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"scores": list(map(float, scores))}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# Dataset preparation utilities (PopQA / Math500 / Hotpot / MMLU)
# ============================================================

def tensor_to_score_df(scores_tensor: torch.Tensor, methods: List[str]) -> pd.DataFrame:
    arr = scores_tensor.detach().cpu().numpy()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {arr.shape}")
    if arr.shape[1] < len(methods):
        raise ValueError(f"Tensor has {arr.shape[1]} cols, need at least {len(methods)}")
    arr = arr[:, :len(methods)].astype("float32", copy=False)
    return pd.DataFrame(arr, columns=methods)


def attach_scores_from_pt_paths(df: pd.DataFrame, pt_paths: List[str], methods: List[str]) -> pd.DataFrame:
    parts = []
    for p in pt_paths:
        t = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(t, dict) and "scores" in t:
            t = t["scores"]
        parts.append(tensor_to_score_df(t, methods))
    scores_all = pd.concat(parts, axis=0, ignore_index=True)
    if len(scores_all) != len(df):
        raise ValueError(f"Scores rows {len(scores_all)} != df rows {len(df)}. Row-order mismatch?")
    out = df.copy()
    for m in methods:
        out[m] = scores_all[m].to_numpy()
    return out


def prepare_popqa_data_from_paths(
    train_csv_path: str,
    test_csv_path: str,
    train_scores_pt_path: str,
    test_scores_pt_paths: List[str],
    methods: Optional[List[str]] = None,
    drop_overlapping_questions: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if methods is None:
        methods = ["MaximumSequenceProbability", "Perplexity", "MeanTokenEntropy"]

    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    df_train = attach_scores_from_pt_paths(df_train, [train_scores_pt_path], methods)
    df_test = attach_scores_from_pt_paths(df_test, test_scores_pt_paths, methods)

    # Normalize answers (train/test)

    df_train["final_answer_init"] = df_train["final_answer_new"]
    df_test["final_answer_init"] = df_test["final_answer_new"]

    df_train["final_answer_new"] = df_train["final_answer_new"].map(normalize_answer_simple)
    df_test["final_answer_new"] = df_test["final_answer_new"].map(normalize_answer_simple)

    # row_hit
    df_train["possible_list"] = df_train["possible_answers"].apply(to_list_safe)
    df_train["row_hit"] = [hit_substring(p, lst) for p, lst in zip(df_train["final_answer_new"], df_train["possible_list"])]

    df_test["possible_list"] = df_test["possible_answers"].apply(to_list_safe)
    df_test["row_hit"] = [hit_substring(p, lst) for p, lst in zip(df_test["final_answer_new"], df_test["possible_list"])]

    # answer_id
    df_train["answer_id"] = df_train.groupby("id")["final_answer_new"].transform(lambda x: pd.factorize(x)[0]).astype(int)
    df_test["answer_id"] = df_test.groupby("id")["final_answer_new"].transform(lambda x: pd.factorize(x)[0]).astype(int)

    if drop_overlapping_questions and "question" in df_train.columns and "question" in df_test.columns:
        overlap = set(df_train["question"]).intersection(set(df_test["question"]))
        df_test = df_test[~df_test["question"].isin(overlap)].copy()

    return df_train, df_test


def prepare_math500_data_from_paths(
    train_csv_path: str,
    test_csv_path: str,
    *,
    unique_id_col: str = "unique_id",
    id_col: str = "id",
    answer_col: str = "final_answer_new",
    rationale_col: str = "rationale",
    completion_col: str = "completion",
    make_answer_id: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # stable ids in each split
    df_train[id_col] = pd.factorize(df_train[unique_id_col])[0]
    df_test[id_col] = pd.factorize(df_test[unique_id_col])[0]

    # fill + normalize
    df_train[answer_col] = df_train[answer_col].fillna("")
    # .map(normalize_answer_simple)
    df_test[answer_col] = df_test[answer_col].fillna("")
    # .map(normalize_answer_simple)

    # rationale fallback
    if rationale_col in df_train.columns:
        if completion_col in df_train.columns:
            df_train[rationale_col] = df_train[rationale_col].fillna(df_train[completion_col])
        else:
            df_train[rationale_col] = df_train[rationale_col].fillna("")
    else:
        df_train[rationale_col] = df_train[completion_col].fillna("") if completion_col in df_train.columns else ""

    if rationale_col in df_test.columns:
        if completion_col in df_test.columns:
            df_test[rationale_col] = df_test[rationale_col].fillna(df_test[completion_col])
        else:
            df_test[rationale_col] = df_test[rationale_col].fillna("")
    else:
        df_test[rationale_col] = df_test[completion_col].fillna("") if completion_col in df_test.columns else ""

    # answer_id (train only, usually)
    if make_answer_id:
        df_train["answer_id"] = df_train.groupby(id_col)[answer_col].transform(lambda x: pd.factorize(x)[0]).astype(int)

    return df_train, df_test


def prepare_mmlu_data_from_paths(
    csv_path: str,
    *,
    question_id_col: str = "question_id",
    id_col: str = "id",
    answer_col: str = "final_answer_new",
    rationale_col: str = "rationale",
    completion_col: str = "completion",
    test_size_ids: int = 5000,
    seed: int = 42,
    make_answer_id: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)

    # id from question_id
    df[id_col] = pd.factorize(df[question_id_col])[0]

    # normalize answers
    df[answer_col] = df[answer_col].fillna("").map(normalize_answer_simple)

    # rationale fallback
    if rationale_col in df.columns and completion_col in df.columns:
        df[rationale_col] = df[rationale_col].fillna(df[completion_col])
    elif rationale_col not in df.columns:
        df[rationale_col] = df[completion_col].fillna("") if completion_col in df.columns else ""

    # answer_id within id
    if make_answer_id:
        df["answer_id"] = df.groupby(id_col)[answer_col].transform(lambda x: pd.factorize(x)[0]).astype(int)

    # split ids for test
    rng = np.random.default_rng(seed)
    unique_ids = df[id_col].unique()
    k = min(int(test_size_ids), len(unique_ids))
    sample_ids = rng.choice(unique_ids, size=k, replace=False)

    mask = df[id_col].isin(sample_ids)
    df_test = df[mask].copy()
    df_train = df[~mask].copy()

    return df_train, df_test


# ============================================================
# Majority vote baseline (optional)
# ============================================================

def majority_hit_group(g: pd.DataFrame) -> bool:
    mode_series = g["final_answer_new"].mode()
    maj = mode_series.iloc[0] if not mode_series.empty else g["final_answer_new"].iloc[0]
    union_possible = [p for sub in g.get("possible_list", []) for p in (sub if isinstance(sub, list) else [])]
    return hit_substring(maj, union_possible)


def majority_baseline_acc_over_topn(
    df: pd.DataFrame,
    topn_values: Iterable[int],
    id_col: str = "id",
    show_progress: bool = True,
) -> List[float]:
    out = []
    it = tqdm(list(topn_values), desc="Majority baseline", unit="top_n") if show_progress else topn_values
    for n in it:
        df_top = df.groupby(id_col, sort=False).head(int(n))
        acc = df_top.groupby(id_col, sort=False).apply(majority_hit_group).mean()
        out.append(float(acc))
        if show_progress:
            it.set_postfix(top_n=int(n), acc=float(acc))
    return out