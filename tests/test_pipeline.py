"""Comprehensive test suite for Multimodal Price Predictor."""

import numpy as np
import pandas as pd
import pytest

from src.price_predictor.data import (
    align_by_sample_id,
    build_submission,
    load_pipeline_config,
    ordered_id_hash,
    save_npy,
    validate_frame,
    validate_npy_artifact,
)
from src.price_predictor.features import (
    CATALOGUE_FEATURE_NAMES,
    FEATURE_BLOCKS,
    FINAL_FEATURE_DIM,
    CatalogFeatureExtractor,
    MiniLMEncoder,
    Siglip2Encoder,
    assemble_feature_artifacts,
    assemble_feature_matrix,
    cosine_similarity_column,
    extract_measurement,
    extract_pack_quantity,
    write_feature_block,
)
from src.price_predictor.models import (
    MODEL_ORDER,
    _feature_matrix,
    blend_predictions,
    create_shared_folds,
    load_fold_assignments,
    optimize_smape_weights,
    save_fold_assignments,
    smape,
    validate_shared_folds,
)
from src.price_predictor.pipeline import MultimodalPricePredictorPipeline


def test_canonical_config_and_feature_offsets():
    config, digest = load_pipeline_config("configs/final.json")
    assert config["feature_width"] == FINAL_FEATURE_DIM == 2450
    assert [block[2] - block[1] for block in FEATURE_BLOCKS] == [1024, 1024, 1, 384, 17]
    assert [block["name"] for block in config["feature_blocks"]] == [
        block[0] for block in FEATURE_BLOCKS
    ]
    assert config["model_order"] == list(MODEL_ORDER)
    assert len(digest) == 64


def test_catalogue_feature_extraction_and_rules():
    train_series = pd.Series([
        "Item Name: Premium Coffee\nBrand: AcmeRoast\nDescription: Dark roast\n1.5 kg pack of 2",
        "Item Name: Matcha Tea\nBrand: AcmeRoast\nProduct Class: Beverages\n500 g",
    ])
    extractor = CatalogFeatureExtractor()
    matrix = extractor.fit_transform(train_series)
    
    unseen = extractor.transform(pd.Series(["Brand: BrandNew\nProduct Class: Other\n250 ml"]))
    
    assert matrix.shape == (2, 17)
    assert matrix.columns.tolist() == list(CATALOGUE_FEATURE_NAMES)
    assert matrix.loc[0, "pack_quantity"] == 2.0
    assert matrix.loc[0, "base_unit_value"] == 1500.0  # 1.5 kg -> 1500 g
    assert matrix.loc[0, "total_declared_value"] == 3000.0
    assert unseen.loc[0, "brand_frequency"] == 0.0
    assert unseen.loc[0, "product_class_frequency"] == 0.0
    
    assert extract_pack_quantity("six bars, pack of 6") == 6.0
    assert extract_measurement("500 millilitres") == (500.0, "volume")


def test_embedding_dimension_guards():
    with pytest.raises(ValueError, match="1024"):
        Siglip2Encoder(expected_dim=768)
    with pytest.raises(ValueError, match="384"):
        MiniLMEncoder(expected_dim=512)

    image = np.zeros((2, 1024), dtype=np.float32)
    text = np.zeros((2, 1024), dtype=np.float32)
    text[:, 0] = 1.0
    cosine = cosine_similarity_column(image, text)
    assert cosine.shape == (2, 1)
    np.testing.assert_array_equal(cosine, 0.0)


def test_feature_matrix_assembly():
    blocks = {
        "siglip2_image": np.full((2, 1024), 1.0, dtype=np.float32),
        "siglip2_text": np.full((2, 1024), 2.0, dtype=np.float32),
        "siglip2_cosine": np.full((2, 1), 3.0, dtype=np.float32),
        "minilm_text": np.full((2, 384), 4.0, dtype=np.float32),
        "catalogue": np.full((2, 17), 5.0, dtype=np.float32),
    }
    matrix = assemble_feature_matrix(**blocks)
    assert matrix.shape == (2, 2450)
    
    for (_, start, stop), expected in zip(FEATURE_BLOCKS, range(1, 6)):
        np.testing.assert_array_equal(matrix[:, start:stop], expected)

    blocks["catalogue"] = np.zeros((2, 18), dtype=np.float32)
    with pytest.raises(ValueError, match="17"):
        assemble_feature_matrix(**blocks)


def test_feature_sidecars_and_manifests(tmp_path):
    ids = ["sample-1", "sample-2"]
    paths = {}
    widths = dict(zip([item[0] for item in FEATURE_BLOCKS], [1024, 1024, 1, 384, 17]))
    
    for name, width in widths.items():
        path = tmp_path / f"{name}.npy"
        write_feature_block(
            path,
            np.zeros((2, width), dtype=np.float32),
            ids,
            block_name=name,
            config_hash="config-123",
            source_hash="source-123",
        )
        paths[name] = path

    matrix, manifest = assemble_feature_artifacts(
        paths, ids, config_hash="config-123", source_hash="source-123"
    )
    assert matrix.shape == (2, 2450)
    assert manifest["ordered_id_hash"] == ordered_id_hash(ids)

    with pytest.raises(ValueError, match="Ordered sample IDs"):
        assemble_feature_artifacts(
            paths, list(reversed(ids)), config_hash="config-123", source_hash="source-123"
        )


def test_shared_folds_and_splits(tmp_path):
    ids = [f"sample-{i}" for i in range(25)]
    folds = create_shared_folds(ids, n_splits=5, random_state=42)
    validate_shared_folds(folds, ids)
    
    assert sorted(np.bincount(folds.folds).tolist()) == [5, 5, 5, 5, 5]
    
    path = tmp_path / "folds.csv"
    save_fold_assignments(folds, path, config_hash="test")
    restored = load_fold_assignments(path, expected_ids=ids, config_hash="test")
    np.testing.assert_array_equal(restored.folds, folds.folds)


def test_smape_optimization_and_blend():
    target = np.array([10.0, 20.0, 30.0, 40.0])
    predictions = np.column_stack([
        target,
        target * 1.1,
        target * 0.9,
        target + 2.0,
    ])
    weights = optimize_smape_weights(predictions, target, minimum_weight=0.01)
    assert weights.shape == (4,)
    assert np.all(weights >= 0.01 - 1e-8)
    assert weights.sum() == pytest.approx(1.0)

    blended = blend_predictions(predictions, weights)
    assert np.all(blended > 0)
    assert smape(target, blended) <= smape(target, predictions.mean(axis=1))


def test_schema_and_submission_contracts():
    invalid_train = pd.DataFrame({
        "sample_id": [1], "catalog_content": ["Test"], "image_link": ["http://img.jpg"], "price": [-5.0]
    })
    with pytest.raises(ValueError, match="greater than zero"):
        validate_frame(invalid_train, training=True)

    sub = build_submission([101, 102], [25.5, 49.99])
    assert sub.columns.tolist() == ["sample_id", "price"]
    assert len(sub) == 2


def test_pipeline_single_prediction():
    pipeline = MultimodalPricePredictorPipeline()
    res = pipeline.predict_single(
        "Item Name: Whole Bean Coffee\nBrand: Roaster\nPack of 2, 500 g each."
    )
    assert "predicted_price" in res
    assert res["predicted_price"] > 0
    assert "model_breakdown" in res
    assert len(res["model_breakdown"]) == 4
    assert res["pack_quantity"] == 2
