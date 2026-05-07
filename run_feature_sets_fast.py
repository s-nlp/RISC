#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd

import lightgbm as lgb

from SC_all_datasets_checkpoints_optimised_refine_fast import (
    prepare_popqa_data_from_paths,
    prepare_math500_data_from_paths,
    filter_questions_for_reranker,
)

from SC_all_datasets_checkpoints_optimised_refine_fast import (
    FeatureConfig,
    EmbeddingCaches,
    train_lgbm_reranker_multibudget,
    train_lgbm_reranker_multibudget_with_hparam_search,
    evaluate_over_budgets,
    save_scores_list,
)


# =========================================================
# Dataset configs
# =========================================================

DATASET_CONFIGS = {
    "popqa": {
        "loader": "popqa",
        "train_csv_path": "/home/jovyan/shares/SR003.nfs2/lysyuk/generate_ASC/longft/data_expanded/popqa_train_expanded.csv",
        "test_csv_path": "/home/jovyan/shares/SR003.nfs2/lysyuk/generate_ASC/longft/data_expanded/popqa_test_expanded.csv",
        "train_scores_pt_path": "self-consistency/popqa_full_train_meta-llama_Meta-Llama-3.1-8B-Instruct_chat.pt",
        "test_scores_pt_paths": [
            "/home/jovyan/shares/SR003.nfs2/lysyuk/huawei_hallu_detector/LongFT/self-consistency/popqa_full_part1_meta-llama_Meta-Llama-3.1-8B-Instruct_chat.pt",
            "/home/jovyan/shares/SR003.nfs2/lysyuk/huawei_hallu_detector/LongFT/self-consistency/popqa_full_part2_meta-llama_Meta-Llama-3.1-8B-Instruct_chat.pt",
        ],
        "min_rows_per_id": 90,
        "cache_path": "caches/embedding_caches_popqa.joblib",
    },
    "math500": {
        "loader": "math500",
        "train_csv_path": "/home/jovyan/shares/SR003.nfs2/lysyuk/generate_ASC/longft/data_expanded/math500_train_expanded.csv",
        "test_csv_path": "/home/jovyan/shares/SR003.nfs2/lysyuk/generate_ASC/longft/data_expanded/math500_test_expanded.csv",
        "cache_path": "caches/embedding_caches_math500.joblib",
    },
    "hotpotqa": {
        "loader": "hotpotqa",
        "train_csv_path": "/home/jovyan/shares/SR003.nfs2/lysyuk/generate_ASC/longft/data_expanded/hotpotqa_train_expanded.csv",
        "test_csv_path": "/home/jovyan/shares/SR003.nfs2/lysyuk/generate_ASC/longft/data_expanded/hotpotqa_test_expanded.csv",
        "unique_id_col": "id",
        "train_max_unique_ids": 8000,
        "cache_path": "caches/embedding_caches_hotpotqa.joblib",
    },
}


# =========================================================
# Feature sets
# =========================================================

FEATURE_SET_CONFIGS = {

    "set7_hyp_search_adaptive": {
    "handcrafted_cols": [
        "share_ratio_to_best",
        "ans_len_min",
    ],
    "embedding_cols": [
        "ans_dist2_to_id_centroid",
        "step_to_chain_centroid_min"
    ],
    "prefix_embedding_cols": [
        "shared_checkpoint_count",
    ]},

    "set7_hyp_search_adaptive": {
    "handcrafted_cols": [
        "share_ratio_to_best",
        "ans_len_min",
    ],
    "embedding_cols": [
        "ans_dist2_to_id_centroid",
        "step_to_chain_centroid_min"
    ],
    "prefix_embedding_cols": [
        "shared_checkpoint_count",
    ]},

    "set7_ablation1": {
    "handcrafted_cols": [
        "ans_len_min",
    ],
    "embedding_cols": [
        "ans_dist2_to_id_centroid",
        "step_to_chain_centroid_min"
    ],
    "prefix_embedding_cols": [
        "shared_checkpoint_count",
    ]},

    "set7_ablation2": {
    "handcrafted_cols": [
        "share_ratio_to_best",
    ],
    "embedding_cols": [
        "ans_dist2_to_id_centroid",
        "step_to_chain_centroid_min"
    ],
    "prefix_embedding_cols": [
        "shared_checkpoint_count",
    ]},

    "set7_ablation3": {
    "handcrafted_cols": [
        "share_ratio_to_best",
        "ans_len_min",
    ],
    "embedding_cols": [
        "step_to_chain_centroid_min"
    ],
    "prefix_embedding_cols": [
        "shared_checkpoint_count",
    ]},

    "set7_ablation4": {
    "handcrafted_cols": [
        "share_ratio_to_best",
        "ans_len_min",
    ],
    "embedding_cols": [
        "ans_dist2_to_id_centroid",
    ],
    "prefix_embedding_cols": [
        "shared_checkpoint_count",
    ]},

    "set7_ablation5": {
    "handcrafted_cols": [
        "share_ratio_to_best",
        "ans_len_min",
    ],
    "embedding_cols": [
        "ans_dist2_to_id_centroid",
        "step_to_chain_centroid_min"
    ],
    "prefix_embedding_cols": [
    ]},

    

    "set7_hyp_search_sim08_rel05": {
    "handcrafted_cols": [
        "share_ratio_to_best",
        "ans_len_min",
    ],
    "embedding_cols": [
        "ans_dist2_to_id_centroid",
        "step_to_chain_centroid_min"
    ],
    "prefix_embedding_cols": [
        "shared_checkpoint_count",
    ]},

    
    "set7_hyp_search_sim088_rel05": {
    "handcrafted_cols": [
        "share_ratio_to_best",
        "ans_len_min",
    ],
    "embedding_cols": [
        "ans_dist2_to_id_centroid",
        "step_to_chain_centroid_min"
    ],
    "prefix_embedding_cols": [
        "shared_checkpoint_count",
    ]},

    "set7_hyp_search_sim075_rel05": {
    "handcrafted_cols": [
        "share_ratio_to_best",
        "ans_len_min",
    ],
    "embedding_cols": [
        "ans_dist2_to_id_centroid",
        "step_to_chain_centroid_min"
    ],
    "prefix_embedding_cols": [
        "shared_checkpoint_count",
    ]},

   
}


# =========================================================
# LightGBM params
# =========================================================

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

# PARAM_GRID = {
#     "n_estimators": [500, 600, 800],
#     "learning_rate": [0.003, 0.005, 0.01],
#     "num_leaves": [16, 20, 31],
#     "max_depth": [4, 5, 6],
#     "min_data_in_leaf": [10, 20, 40],
#     "reg_lambda": [0.3, 1.0, 3.0],
#     "feature_fraction_bynode": [0.6, 0.8, 1.0],
# }

PARAM_GRID = {
    "n_estimators": [500, 600, 700],
    "learning_rate": [0.005, 0.01],
    "num_leaves": [24, 31],
    "max_depth": [5, 6],
    "min_data_in_leaf": [20, 40],
    "reg_lambda": [0.3, 1.0],
    "feature_fraction_bynode": [0.8, 1.0],
}

# =========================================================
# Helpers
# =========================================================

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


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
        prefix_rel_support_thr=(fs.get("prefix_rel_support_thr", 1.0 / 3.0) if prefix_rel_support_thr is None else float(prefix_rel_support_thr)),
        adaptive_prefix_thresholds=bool(adaptive_prefix_thresholds),
        batch_size=batch_size,
        device=device,
    )


def filter_popqa_split_by_id_count(df, min_rows_per_id: int):
    rows_per_id = df.groupby("id").size().reset_index(name="n_rows")
    ids_to_keep = rows_per_id.loc[rows_per_id["n_rows"] > min_rows_per_id, "id"]
    df_filtered = df[df["id"].isin(ids_to_keep)].copy()
    return df_filtered


def load_popqa(cfg: dict):
    df_train, df_test = prepare_popqa_data_from_paths(
        train_csv_path=cfg["train_csv_path"],
        test_csv_path=cfg["test_csv_path"],
        train_scores_pt_path=cfg["train_scores_pt_path"],
        test_scores_pt_paths=cfg["test_scores_pt_paths"],
    )

    df_train = filter_popqa_split_by_id_count(
        df_train,
        min_rows_per_id=cfg["min_rows_per_id"],
    )
    df_test = filter_popqa_split_by_id_count(
        df_test,
        min_rows_per_id=cfg["min_rows_per_id"],
    )

    return df_train, df_test


def load_math500(cfg: dict):
    df_train, df_test = prepare_math500_data_from_paths(
        train_csv_path=cfg["train_csv_path"],
        test_csv_path=cfg["test_csv_path"],
    )
    return df_train, df_test


def load_hotpotqa(cfg: dict):
    df_train, df_test = prepare_math500_data_from_paths(
        train_csv_path=cfg["train_csv_path"],
        test_csv_path=cfg["test_csv_path"],
        unique_id_col=cfg["unique_id_col"],
    )

    first_8000_ids = df_train["id"].drop_duplicates().head(cfg["train_max_unique_ids"])
    df_train = df_train[df_train["id"].isin(first_8000_ids)].copy()

    return df_train, df_test


def load_dataset(dataset_name: str):
    cfg = DATASET_CONFIGS[dataset_name]

    if cfg["loader"] == "popqa":
        df_train, df_test = load_popqa(cfg)
    elif cfg["loader"] == "math500":
        df_train, df_test = load_math500(cfg)
    elif cfg["loader"] == "hotpotqa":
        df_train, df_test = load_hotpotqa(cfg)
    else:
        raise ValueError(f"Unknown loader type: {cfg['loader']}")

    df_train_pruned, _, per_q_stats = filter_questions_for_reranker(df_train)
    return df_train_pruned, df_test, per_q_stats


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--feature-set", required=True, choices=sorted(FEATURE_SET_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 5, 10, 20, 40, 60, 80, 100])
    parser.add_argument("--eval-min", type=int, default=1)
    parser.add_argument("--eval-max", type=int, default=100)  # exclusive
    parser.add_argument("--reranker-dir", default="rerankers")
    parser.add_argument("--scores-dir", default="reranker_scores")
    parser.add_argument("--metadata-dir", default="reranker_metadata")
    parser.add_argument("--no-hparam-search", action="store_true")
    parser.add_argument("--adaptive-prefix-thresholds", action="store_true")
    parser.add_argument("--prefix-sim-threshold", type=float, default=None)
    parser.add_argument("--prefix-rel-support-thr", type=float, default=None)
    args = parser.parse_args()

    dataset = args.dataset
    feature_set = args.feature_set
    ds_cfg = DATASET_CONFIGS[dataset]

    reranker_dir = Path(args.reranker_dir)
    scores_dir = Path(args.scores_dir)
    metadata_dir = Path(args.metadata_dir)
    ensure_dirs(reranker_dir, scores_dir, metadata_dir)

    cache_path = Path(ds_cfg["cache_path"])
    reranker_path = reranker_dir / f"{dataset}_{feature_set}.joblib"
    scores_path = scores_dir / f"{dataset}_{feature_set}.json"
    metadata_path = metadata_dir / f"{dataset}_{feature_set}.json"

    search_results_path = metadata_dir / f"{dataset}_{feature_set}_search_results.csv"
    best_params_path = metadata_dir / f"{dataset}_{feature_set}_best_params.json"

    print("=" * 80)
    print(f"Dataset       : {dataset}")
    print(f"Feature set   : {feature_set}")
    print(f"Cache path    : {cache_path}")
    print(f"Reranker path : {reranker_path}")
    print(f"Scores path   : {scores_path}")
    print("=" * 80)

    df_train_pruned, df_test, per_q_stats = load_dataset(dataset)

    print(f"Train pruned shape : {df_train_pruned.shape}")
    print(f"Test shape         : {df_test.shape}")

    feature_cfg = make_feature_cfg(
        feature_set_name=feature_set,
        device=args.device,
        batch_size=args.batch_size,
        prefix_sim_threshold=args.prefix_sim_threshold,
        prefix_rel_support_thr=args.prefix_rel_support_thr,
        adaptive_prefix_thresholds=args.adaptive_prefix_thresholds,
    )

    if cache_path.exists():
        print(f"Loading caches from {cache_path}")
        caches = EmbeddingCaches.load(str(cache_path))
    else:
        print(f"No cache found at {cache_path}; starting with empty caches")
        caches = EmbeddingCaches()

    if args.no_hparam_search:
        bundle, importances = train_lgbm_reranker_multibudget(
            df_train_trace=df_train_pruned,
            budgets=args.budgets,
            feature_cfg=feature_cfg,
            lgbm_ranker_ctor=lgb.LGBMRanker,
            lgbm_params=DEFAULT_LGB_PARAMS,
            caches=caches,
        )
        search_results = pd.DataFrame()
        best_params = dict(DEFAULT_LGB_PARAMS)
    else:
        bundle, importances, search_results, best_params = train_lgbm_reranker_multibudget_with_hparam_search(
            df_train_trace=df_train_pruned,
            budgets=args.budgets,
            feature_cfg=feature_cfg,
            lgbm_ranker_ctor=lgb.LGBMRanker,
            lgbm_params=DEFAULT_LGB_PARAMS,
            param_grid=PARAM_GRID,
            val_size=0.2,
            random_state=42,
            ndcg_at=10,
            caches=caches,
        )

    scores = evaluate_over_budgets(
        bundle,
        df_test,
        range(args.eval_min, args.eval_max),
        caches=caches,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # caches.save(str(cache_path))
    # print(f"Updated caches saved to {cache_path}")

    save_scores_list(scores, str(scores_path))
    bundle.save(str(reranker_path))
    if len(search_results) > 0:
        search_results.to_csv(search_results_path, index=False)
    save_json(best_params_path, best_params)

    metadata = {
    "dataset": dataset,
    "feature_set": feature_set,
    "cache_path": str(cache_path),
    "reranker_path": str(reranker_path),
    "scores_path": str(scores_path),
    "search_results_path": str(search_results_path),
    "best_params_path": str(best_params_path),
    "feature_config": FEATURE_SET_CONFIGS[feature_set],
    "base_lgbm_params": DEFAULT_LGB_PARAMS,
    "best_lgbm_params": best_params,
    "param_grid": PARAM_GRID,
    "no_hparam_search": bool(args.no_hparam_search),
    "adaptive_prefix_thresholds": bool(args.adaptive_prefix_thresholds),
    "manual_prefix_sim_threshold": args.prefix_sim_threshold,
    "manual_prefix_rel_support_thr": args.prefix_rel_support_thr,
    "budgets_train": args.budgets,
    "eval_range": [args.eval_min, args.eval_max],
    "train_shape": list(df_train_pruned.shape),
    "test_shape": list(df_test.shape),
    "per_q_stats_type": str(type(per_q_stats)),}
    
    save_json(metadata_path, metadata)

    print(f"Saved reranker to {reranker_path}")
    print(f"Saved scores to   {scores_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()