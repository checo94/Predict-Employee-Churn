from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modeling import (
    FEATURES,
    find_similar_profiles,
    load_training_data,
    local_sensitivity,
    predict_scores,
    train_model,
    validate_feature_frame,
    what_if_analysis,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "HCM_Employee_Churn.csv"


@pytest.fixture(scope="session")
def training_data() -> pd.DataFrame:
    return load_training_data(DATA_PATH)


@pytest.fixture(scope="session")
def model_bundle(training_data: pd.DataFrame):
    return train_model(training_data)


def test_source_dataset_is_intact(training_data: pd.DataFrame) -> None:
    assert training_data.shape == (14_999, 7)
    assert int(training_data["left"].sum()) == 3_557
    assert training_data["durchschnittliche_monatliche_arbeitszeit"].max() == 310
    assert training_data.loc[:, FEATURES].drop_duplicates().shape[0] == 11_119


def test_validation_accepts_supported_english_columns_and_salary_aliases() -> None:
    frame = pd.DataFrame(
        {
            "satisfaction_level": [73, 44, 81],
            "number_project": [3, 5, 4],
            "average_montly_hours": [175, 240, 190],
            "Work_accident": [0, 1, 0],
            "promotion_last_5years": [0, 0, 1],
            "salary": ["mittel", 1, "HIGH"],
        }
    )

    report = validate_feature_frame(frame)

    assert report.errors == ()
    assert report.warnings == ()
    assert report.features["gehalt"].tolist() == ["medium", "low", "high"]
    assert tuple(report.features.columns) == FEATURES


def test_validation_reports_ranges_types_and_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "zufriedenheitsgrad": [101, None],
            "anzahl_projekte": [3.5, 4],
            "durchschnittliche_monatliche_arbeitszeit": [180, "unbekannt"],
            "arbeitsunfall": [2, 0],
            "foerderung_letzte_5_jahre": [0, 1],
            "gehalt": ["unbekannt", "mittel"],
        }
    )

    report = validate_feature_frame(frame)

    assert len(report.errors) >= 4
    assert any("Zufriedenheitsgrad" in message for message in report.errors)
    assert any("Gehaltsstufe" in message for message in report.errors)
    assert any("fehlende Werte" in message for message in report.warnings)


def test_validation_warns_about_extrapolation() -> None:
    frame = pd.DataFrame(
        {
            "zufriedenheitsgrad": [5],
            "anzahl_projekte": [10],
            "durchschnittliche_monatliche_arbeitszeit": [350],
            "arbeitsunfall": [0],
            "foerderung_letzte_5_jahre": [0],
            "gehalt": ["mittel"],
        }
    )

    report = validate_feature_frame(frame)

    assert report.errors == ()
    assert len(report.warnings) == 3
    assert all("außerhalb der im Training beobachteten" in warning for warning in report.warnings)


def test_model_has_honest_high_quality_out_of_fold_performance(model_bundle) -> None:
    metrics = model_bundle.balanced_metrics

    assert 0.1 < model_bundle.balanced_threshold < 0.8
    assert model_bundle.sensitive_threshold < model_bundle.balanced_threshold
    assert metrics["roc_auc"] > 0.94
    assert metrics["average_precision"] > 0.90
    assert metrics["balanced_accuracy"] > 0.87
    assert metrics["recall"] > 0.82
    assert model_bundle.sensitive_metrics["recall"] >= metrics["recall"]
    assert int(model_bundle.confusion.sum()) == model_bundle.trained_rows


def test_prediction_and_explanation_helpers(
    training_data: pd.DataFrame,
    model_bundle,
) -> None:
    profile = pd.DataFrame(
        [
            {
                "zufriedenheitsgrad": 42,
                "anzahl_projekte": 6,
                "durchschnittliche_monatliche_arbeitszeit": 260,
                "arbeitsunfall": 0,
                "foerderung_letzte_5_jahre": 0,
                "gehalt": "low",
            }
        ]
    )

    score = predict_scores(model_bundle, profile)
    sensitivity = local_sensitivity(model_bundle, profile)
    peers = find_similar_profiles(training_data, profile, neighbors=100)
    scenarios = what_if_analysis(
        model_bundle,
        profile,
        "zufriedenheitsgrad",
        training_data,
        points=15,
    )

    assert score.shape == (1,)
    assert 0 <= score[0] <= 1
    assert len(sensitivity) == len(FEATURES)
    assert np.isfinite(sensitivity["score_change"]).all()
    assert peers["count"] == 100
    assert 0 <= peers["churn_rate"] <= 1
    assert len(scenarios) == 15
    assert scenarios["score"].between(0, 1).all()
