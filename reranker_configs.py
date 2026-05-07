#!/usr/bin/env python3
from SC_all_datasets_checkpoints_optimised_refine_fast import FeatureConfig

DATASET_CONFIGS = {
    "popqa": {"cache_path": "caches_olmo/popqa_embedding_caches.joblib"},
    "math500": {"cache_path": "caches_olmo/math500_embedding_caches.joblib"},
    "hotpotqa": {"cache_path": "caches_olmo/hotpotqa_embedding_caches.joblib"},
}

FEATURE_SET_CONFIGS = {
    "set7_hyp_search_adaptive": {
        "handcrafted_cols": ["share_ratio_to_best", "ans_len_min"],
        "embedding_cols": ["ans_dist2_to_id_centroid", "step_to_chain_centroid_min"],
        "prefix_embedding_cols": ["shared_checkpoint_count"],
    },
    "set7_ablation1": {
        "handcrafted_cols": ["ans_len_min"],
        "embedding_cols": ["ans_dist2_to_id_centroid", "step_to_chain_centroid_min"],
        "prefix_embedding_cols": ["shared_checkpoint_count"],
    },
    "set7_ablation2": {
        "handcrafted_cols": ["share_ratio_to_best"],
        "embedding_cols": ["ans_dist2_to_id_centroid", "step_to_chain_centroid_min"],
        "prefix_embedding_cols": ["shared_checkpoint_count"],
    },
    "set7_ablation3": {
        "handcrafted_cols": ["share_ratio_to_best", "ans_len_min"],
        "embedding_cols": ["step_to_chain_centroid_min"],
        "prefix_embedding_cols": ["shared_checkpoint_count"],
    },
    "set7_ablation4": {
        "handcrafted_cols": ["share_ratio_to_best", "ans_len_min"],
        "embedding_cols": ["ans_dist2_to_id_centroid"],
        "prefix_embedding_cols": ["shared_checkpoint_count"],
    },
    "set7_ablation5": {
        "handcrafted_cols": ["share_ratio_to_best", "ans_len_min"],
        "embedding_cols": ["ans_dist2_to_id_centroid", "step_to_chain_centroid_min"],
        "prefix_embedding_cols": [],
    },
    "set7_hyp_search_sim08_rel05": {
        "handcrafted_cols": ["share_ratio_to_best", "ans_len_min"],
        "embedding_cols": ["ans_dist2_to_id_centroid", "step_to_chain_centroid_min"],
        "prefix_embedding_cols": ["shared_checkpoint_count"],
    },
    "set7_hyp_search_sim088_rel05": {
        "handcrafted_cols": ["share_ratio_to_best", "ans_len_min"],
        "embedding_cols": ["ans_dist2_to_id_centroid", "step_to_chain_centroid_min"],
        "prefix_embedding_cols": ["shared_checkpoint_count"],
    },
    "set7_hyp_search_sim075_rel05": {
        "handcrafted_cols": ["share_ratio_to_best", "ans_len_min"],
        "embedding_cols": ["ans_dist2_to_id_centroid", "step_to_chain_centroid_min"],
        "prefix_embedding_cols": ["shared_checkpoint_count"],
    },
}

DEFAULT_LGB_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=600,
    learning_rate=0.01,
    max_depth=6,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=1.0,
    min_data_in_leaf=40,
    feature_fraction_bynode=0.8,
    reg_lambda=0.3,
    reg_alpha=1e-3,
    random_state=42,
)

PARAM_GRID = {
    "n_estimators": [500, 600, 700],
    "learning_rate": [0.005, 0.01],
    "num_leaves": [24, 31],
    "max_depth": [5, 6],
    "min_data_in_leaf": [20, 40],
    "reg_lambda": [0.3, 1.0],
    "feature_fraction_bynode": [0.8, 1.0],
}


def make_feature_cfg(
    feature_set_name: str,
    device: str,
    batch_size: int,
    prefix_sim_threshold: float | None = None,
    prefix_rel_support_thr: float | None = None,
    adaptive_prefix_thresholds: bool = False,
) -> FeatureConfig:
    fs = FEATURE_SET_CONFIGS[feature_set_name]
    return FeatureConfig(
        handcrafted_cols=fs["handcrafted_cols"],
        use_embeddings=len(fs["embedding_cols"]) > 0,
        embedding_cols=fs["embedding_cols"],
        use_prefix_embeddings=len(fs["prefix_embedding_cols"]) > 0,
        prefix_embedding_cols=fs["prefix_embedding_cols"],
        prefix_mode="step",
        prefix_word_stride=10,
        prefix_min_words=5,
        prefix_sim_threshold=0.8 if prefix_sim_threshold is None else float(prefix_sim_threshold),
        prefix_depth_tol=0.20,
        prefix_early_frac=0.3,
        prefix_late_frac=0.7,
        prefix_rel_support_thr=(
            fs.get("prefix_rel_support_thr", 1.0 / 3.0)
            if prefix_rel_support_thr is None
            else float(prefix_rel_support_thr)
        ),
        adaptive_prefix_thresholds=bool(adaptive_prefix_thresholds),
        batch_size=batch_size,
        device=device,
    )
