"""Command-line interface for Multimodal Price Predictor workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from .data import (
    build_submission,
    load_pipeline_config,
    save_npy,
    validate_frame,
)
from .features import (
    CatalogFeatureExtractor,
    MiniLMEncoder,
    Siglip2Encoder,
    assemble_feature_matrix,
    cosine_similarity_column,
)
from .models import (
    CatBoostModel,
    LightGBMModel,
    PyTorchPriceMLP,
    XGBoostModel,
    blend_predictions,
    create_shared_folds,
    optimize_smape_weights,
    smape,
)


def validate_data_command(args):
    print(f"Validating training CSV: {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)
    validate_frame(train_df, training=True)
    print(f"✅ Training data valid! ({len(train_df)} rows)")

    if args.test_csv:
        print(f"Validating test CSV: {args.test_csv}")
        test_df = pd.read_csv(args.test_csv)
        validate_frame(test_df, training=False)
        print(f"✅ Test data valid! ({len(test_df)} rows)")


def run_full_pipeline_command(args):
    print("🚀 Starting Multimodal Price Predictor Pipeline...")
    config, digest = load_pipeline_config(args.config)
    print(f"Loaded config: {args.config} (digest: {digest[:8]}...)")

    train_df = pd.read_csv(args.train_csv)
    validate_frame(train_df, training=True)
    print(f"Train dataset loaded: {len(train_df)} samples")

    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature extraction
    print("Extracting SigLIP2 and MiniLM multimodal embeddings...")
    siglip = Siglip2Encoder(device=args.device)
    minilm = MiniLMEncoder(device=args.device)
    catalog_ext = CatalogFeatureExtractor()

    sig_img = siglip.encode_images([None] * len(train_df))
    sig_txt = siglip.encode_text(train_df["catalog_content"])
    sig_cos = cosine_similarity_column(sig_img, sig_txt)
    mini_txt = minilm.encode(train_df["catalog_content"])
    cat_feats = catalog_ext.fit_transform(train_df["catalog_content"]).to_numpy(dtype=np.float32)

    X_train = assemble_feature_matrix(sig_img, sig_txt, sig_cos, mini_txt, cat_feats)
    y_train = train_df["price"].to_numpy(dtype=np.float32)
    y_log = np.log1p(y_train)

    print(f"Feature matrix assembled: {X_train.shape}")
    save_npy(out_dir / "X_train.npy", X_train)

    # 2. 5-Fold Ensemble Training
    print("Training 5-Fold Ensemble (LightGBM, XGBoost, CatBoost, PyTorch MLP)...")
    folds = create_shared_folds(train_df["sample_id"], n_splits=5)
    oof_preds = np.zeros((len(train_df), 4), dtype=np.float64)

    for f_idx, (trn_idx, val_idx) in enumerate(folds.split()):
        print(f"  Training Fold {f_idx + 1}/5...")
        X_tr, y_tr = X_train[trn_idx], y_log[trn_idx]
        X_va, y_va = X_train[val_idx], y_log[val_idx]

        m_lgb = LightGBMModel().fit(X_tr, y_tr)
        m_xgb = XGBoostModel().fit(X_tr, y_tr)
        m_cat = CatBoostModel().fit(X_tr, y_tr)
        m_mlp = PyTorchPriceMLP(in_features=X_train.shape[1]).fit(X_tr, y_tr, epochs=5, device=args.device)

        oof_preds[val_idx, 0] = np.expm1(m_lgb.predict(X_va))
        oof_preds[val_idx, 1] = np.expm1(m_xgb.predict(X_va))
        oof_preds[val_idx, 2] = np.expm1(m_cat.predict(X_va))
        oof_preds[val_idx, 3] = np.expm1(m_mlp.predict(X_va, device=args.device))

    # 3. SMAPE Blend Optimization
    weights = optimize_smape_weights(oof_preds, y_train)
    blended_oof = blend_predictions(oof_preds, weights)
    final_smape = smape(y_train, blended_oof)

    print(f"🏆 Optimized Ensemble SMAPE: {final_smape:.2f}%")
    print(f"Learned Weights: LGB={weights[0]:.3f}, XGB={weights[1]:.3f}, CAT={weights[2]:.3f}, MLP={weights[3]:.3f}")

    # 4. Predict on Test set if provided
    if args.test_csv:
        test_df = pd.read_csv(args.test_csv)
        print(f"Generating predictions for {len(test_df)} test samples...")
        
        sig_img_te = siglip.encode_images([None] * len(test_df))
        sig_txt_te = siglip.encode_text(test_df["catalog_content"])
        sig_cos_te = cosine_similarity_column(sig_img_te, sig_txt_te)
        mini_txt_te = minilm.encode(test_df["catalog_content"])
        cat_feats_te = catalog_ext.transform(test_df["catalog_content"]).to_numpy(dtype=np.float32)

        X_test = assemble_feature_matrix(sig_img_te, sig_txt_te, sig_cos_te, mini_txt_te, cat_feats_te)
        
        # Fit on full data for test inference
        m_lgb_full = LightGBMModel().fit(X_train, y_log)
        m_xgb_full = XGBoostModel().fit(X_train, y_log)
        m_cat_full = CatBoostModel().fit(X_train, y_log)
        m_mlp_full = PyTorchPriceMLP(in_features=X_train.shape[1]).fit(X_train, y_log, epochs=5, device=args.device)

        test_preds = np.column_stack([
            np.expm1(m_lgb_full.predict(X_test)),
            np.expm1(m_xgb_full.predict(X_test)),
            np.expm1(m_cat_full.predict(X_test)),
            np.expm1(m_mlp_full.predict(X_test, device=args.device)),
        ])
        blended_test = blend_predictions(test_preds, weights)
        sub_df = build_submission(test_df["sample_id"], blended_test)
        sub_path = out_dir / "submission.csv"
        sub_df.to_csv(sub_path, index=False)
        print(f"✅ Submission saved to: {sub_path}")


def main():
    parser = argparse.ArgumentParser(description="Multimodal Price Predictor CLI")
    subparsers = parser.add_subparsers(dest="command")

    # validate-data
    val_p = subparsers.add_parser("validate-data")
    val_p.add_argument("--train-csv", required=True)
    val_p.add_argument("--test-csv", default=None)

    # run-pipeline
    run_p = subparsers.add_parser("run")
    run_p.add_argument("--train-csv", required=True)
    run_p.add_argument("--test-csv", default=None)
    run_p.add_argument("--config", default="configs/final.json")
    run_p.add_argument("--artifact-dir", default="artifacts/final")
    run_p.add_argument("--device", default="cpu")

    args = parser.parse_args()
    if args.command == "validate-data":
        validate_data_command(args)
    elif args.command == "run":
        run_full_pipeline_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
