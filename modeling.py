"""Data validation, model training, evaluation, and explainability helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
CV_SPLITS = 5
MODEL_VERSION = "2.0"
TARGET = "left"

FEATURES = (
    "zufriedenheitsgrad",
    "anzahl_projekte",
    "durchschnittliche_monatliche_arbeitszeit",
    "arbeitsunfall",
    "foerderung_letzte_5_jahre",
    "gehalt",
)

NUMERIC_FEATURES = FEATURES[:-1]
CATEGORICAL_FEATURES = ("gehalt",)

DISPLAY_NAMES = {
    "zufriedenheitsgrad": "Zufriedenheitsgrad",
    "anzahl_projekte": "Anzahl Projekte",
    "durchschnittliche_monatliche_arbeitszeit": "Monatliche Arbeitszeit",
    "arbeitsunfall": "Arbeitsunfall",
    "foerderung_letzte_5_jahre": "Förderung in den letzten 5 Jahren",
    "gehalt": "Gehaltsstufe",
}

SOURCE_COLUMN_ALIASES = {
    "satisfaction_level": "zufriedenheitsgrad",
    "number_project": "anzahl_projekte",
    "average_montly_hours": "durchschnittliche_monatliche_arbeitszeit",
    "average_monthly_hours": "durchschnittliche_monatliche_arbeitszeit",
    "Work_accident": "arbeitsunfall",
    "work_accident": "arbeitsunfall",
    "promotion_last_5years": "foerderung_letzte_5_jahre",
    "salary": "gehalt",
    "churn": TARGET,
}

FEATURE_RANGES = {
    "zufriedenheitsgrad": (0.0, 100.0),
    "anzahl_projekte": (1.0, 20.0),
    "durchschnittliche_monatliche_arbeitszeit": (1.0, 744.0),
    "arbeitsunfall": (0.0, 1.0),
    "foerderung_letzte_5_jahre": (0.0, 1.0),
}

TRAINING_RANGES = {
    "zufriedenheitsgrad": (9.0, 100.0),
    "anzahl_projekte": (2.0, 7.0),
    "durchschnittliche_monatliche_arbeitszeit": (96.0, 310.0),
}

INTEGER_FEATURES = {
    "anzahl_projekte",
    "durchschnittliche_monatliche_arbeitszeit",
    "arbeitsunfall",
    "foerderung_letzte_5_jahre",
}

SALARY_ALIASES = {
    "1": "low",
    "1.0": "low",
    "low": "low",
    "niedrig": "low",
    "2": "medium",
    "2.0": "medium",
    "medium": "medium",
    "mittel": "medium",
    "3": "high",
    "3.0": "high",
    "high": "high",
    "hoch": "high",
}

SALARY_DISPLAY = {"low": "Niedrig", "medium": "Mittel", "high": "Hoch"}


class DataValidationError(ValueError):
    """Raised when a training or prediction dataset cannot be used safely."""


@dataclass(frozen=True)
class ValidationReport:
    """Normalized features together with user-facing validation messages."""

    features: pd.DataFrame
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass
class ModelBundle:
    """All reusable artifacts produced by one deterministic training run."""

    pipeline: Pipeline
    balanced_threshold: float
    sensitive_threshold: float
    balanced_metrics: dict[str, float]
    sensitive_metrics: dict[str, float]
    confusion: np.ndarray
    feature_importance: pd.DataFrame
    reference_values: dict[str, Any]
    oof_probabilities: np.ndarray
    oof_predictions: np.ndarray
    trained_rows: int
    unique_profiles: int
    churn_rate: float
    model_version: str = MODEL_VERSION


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with supported English column names mapped to German names."""
    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]

    rename_map: dict[str, str] = {}
    for source, target in SOURCE_COLUMN_ALIASES.items():
        if source not in normalized.columns:
            continue
        if target in normalized.columns and source != target:
            raise DataValidationError(
                f"Die Datei enthält sowohl '{source}' als auch '{target}'. "
                "Bitte verwenden Sie nur eine der beiden Spalten."
            )
        rename_map[source] = target

    normalized = normalized.rename(columns=rename_map)
    if normalized.columns.duplicated().any():
        duplicates = sorted(set(normalized.columns[normalized.columns.duplicated()]))
        raise DataValidationError(
            "Doppelte Spalten nach der Normalisierung: " + ", ".join(duplicates)
        )
    return normalized


def _normalize_salary(value: Any) -> str | float:
    if pd.isna(value):
        return np.nan
    return SALARY_ALIASES.get(str(value).strip().lower(), np.nan)


def _row_numbers(mask: pd.Series, limit: int = 8) -> str:
    positions = np.flatnonzero(mask.to_numpy())[:limit] + 2
    suffix = " …" if int(mask.sum()) > limit else ""
    return ", ".join(str(int(position)) for position in positions) + suffix


def validate_feature_frame(
    frame: pd.DataFrame,
    *,
    allow_missing: bool = True,
) -> ValidationReport:
    """Validate and normalize a prediction dataframe without row-by-row loops."""
    if frame.empty:
        return ValidationReport(frame.copy(), ("Die Datei enthält keine Datensätze.",), ())

    try:
        normalized = canonicalize_columns(frame)
    except DataValidationError as exc:
        return ValidationReport(frame.copy(), (str(exc),), ())

    missing_columns = [feature for feature in FEATURES if feature not in normalized.columns]
    if missing_columns:
        return ValidationReport(
            normalized,
            ("Fehlende Pflichtspalten: " + ", ".join(missing_columns),),
            (),
        )

    result = normalized.loc[:, FEATURES].copy()
    errors: list[str] = []
    warnings: list[str] = []

    for feature in NUMERIC_FEATURES:
        original = result[feature]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_type = original.notna() & converted.isna()
        if invalid_type.any():
            errors.append(
                f"{DISPLAY_NAMES[feature]} ist in Zeile(n) "
                f"{_row_numbers(invalid_type)} keine gültige Zahl."
            )

        minimum, maximum = FEATURE_RANGES[feature]
        outside_range = converted.notna() & ~converted.between(minimum, maximum)
        if outside_range.any():
            errors.append(
                f"{DISPLAY_NAMES[feature]} liegt in Zeile(n) "
                f"{_row_numbers(outside_range)} außerhalb des zulässigen Bereichs "
                f"{minimum:g}–{maximum:g}."
            )

        if feature in TRAINING_RANGES:
            training_minimum, training_maximum = TRAINING_RANGES[feature]
            outside_training = (
                converted.notna()
                & ~outside_range
                & ~converted.between(training_minimum, training_maximum)
            )
            if outside_training.any():
                warnings.append(
                    f"{DISPLAY_NAMES[feature]} liegt in Zeile(n) "
                    f"{_row_numbers(outside_training)} außerhalb der im Training beobachteten "
                    f"Spanne {training_minimum:g}–{training_maximum:g}; der Score ist dort "
                    "besonders unsicher."
                )

        if feature in INTEGER_FEATURES:
            non_integer = converted.notna() & ~np.isclose(converted % 1, 0)
            if non_integer.any():
                errors.append(
                    f"{DISPLAY_NAMES[feature]} muss in Zeile(n) "
                    f"{_row_numbers(non_integer)} ganzzahlig sein."
                )

        result[feature] = converted

    salary_original = result["gehalt"]
    result["gehalt"] = salary_original.map(_normalize_salary)
    invalid_salary = salary_original.notna() & result["gehalt"].isna()
    if invalid_salary.any():
        errors.append(
            "Gehaltsstufe ist in Zeile(n) "
            f"{_row_numbers(invalid_salary)} ungültig. Erlaubt sind niedrig/mittel/hoch, "
            "low/medium/high oder 1/2/3."
        )

    for feature in FEATURES:
        missing_count = int(result[feature].isna().sum())
        if not missing_count:
            continue
        if allow_missing:
            warnings.append(
                f"{DISPLAY_NAMES[feature]}: {missing_count} fehlende Werte werden mit "
                "Trainingswerten imputiert."
            )
        else:
            errors.append(f"{DISPLAY_NAMES[feature]} enthält {missing_count} fehlende Werte.")

    return ValidationReport(result, tuple(errors), tuple(warnings))


def load_training_data(path: str | Path) -> pd.DataFrame:
    """Load the intact source dataset and enforce its modeling contract."""
    source = Path(path)
    if not source.is_file():
        raise DataValidationError(f"Trainingsdatei nicht gefunden: {source.name}")

    raw = pd.read_csv(source)
    canonical = canonicalize_columns(raw)
    if TARGET not in canonical.columns:
        raise DataValidationError(f"Die Zielspalte '{TARGET}' fehlt in der Trainingsdatei.")

    report = validate_feature_frame(canonical, allow_missing=False)
    if report.errors:
        raise DataValidationError(" ".join(report.errors))

    target = pd.to_numeric(canonical[TARGET], errors="coerce")
    invalid_target = target.isna() | ~target.isin((0, 1))
    if invalid_target.any():
        raise DataValidationError(
            f"Die Zielspalte '{TARGET}' enthält ungültige Werte in "
            f"Zeile(n) {_row_numbers(invalid_target)}."
        )

    training_data = report.features.copy()
    training_data[TARGET] = target.astype("int8")
    return training_data


def build_pipeline() -> Pipeline:
    """Build an efficient nonlinear model with robust preprocessing."""
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        verbose_feature_names_out=False,
    )

    classifier = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def _select_threshold(
    target: pd.Series,
    probabilities: np.ndarray,
    *,
    beta: float,
) -> float:
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    beta_squared = beta**2
    scores = (1 + beta_squared) * precision * recall / (beta_squared * precision + recall + 1e-12)
    best_index = int(np.nanargmax(scores[:-1]))
    return float(thresholds[best_index])


def _metrics_at_threshold(
    target: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray]:
    predictions = (probabilities >= threshold).astype("int8")
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity = true_negative / max(true_negative + false_positive, 1)

    metrics = {
        "accuracy": accuracy_score(target, predictions),
        "balanced_accuracy": balanced_accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(target, predictions, zero_division=0),
        "roc_auc": roc_auc_score(target, probabilities),
        "average_precision": average_precision_score(target, probabilities),
        "brier_score": brier_score_loss(target, probabilities),
        "threshold": threshold,
    }
    return metrics, matrix


def train_model(training_data: pd.DataFrame) -> ModelBundle:
    """Train once and evaluate with duplicate-profile-safe out-of-fold predictions."""
    features = training_data.loc[:, FEATURES]
    target = training_data[TARGET].astype("int8")
    groups = pd.util.hash_pandas_object(features, index=False)
    cross_validator = StratifiedGroupKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    evaluation_pipeline = build_pipeline()
    out_of_fold_probabilities = cross_val_predict(
        evaluation_pipeline,
        features,
        target,
        groups=groups,
        cv=cross_validator,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    balanced_threshold = _select_threshold(target, out_of_fold_probabilities, beta=1.0)
    sensitive_threshold = _select_threshold(target, out_of_fold_probabilities, beta=2.0)
    balanced_metrics, matrix = _metrics_at_threshold(
        target, out_of_fold_probabilities, balanced_threshold
    )
    sensitive_metrics, _ = _metrics_at_threshold(
        target, out_of_fold_probabilities, sensitive_threshold
    )

    # Model-agnostic importance is measured on a group-isolated fold so identical
    # profiles never appear in both the fitting and evaluation partitions.
    train_indices, test_indices = next(cross_validator.split(features, target, groups=groups))
    importance_pipeline = build_pipeline()
    importance_pipeline.fit(features.iloc[train_indices], target.iloc[train_indices])
    importance = permutation_importance(
        importance_pipeline,
        features.iloc[test_indices],
        target.iloc[test_indices],
        scoring="average_precision",
        n_repeats=6,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    feature_importance = pd.DataFrame(
        {
            "feature": list(FEATURES),
            "label": [DISPLAY_NAMES[feature] for feature in FEATURES],
            "importance": importance.importances_mean,
            "standard_deviation": importance.importances_std,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)

    final_pipeline = build_pipeline()
    final_pipeline.fit(features, target)

    reference_values: dict[str, Any] = {
        feature: float(features[feature].median()) for feature in NUMERIC_FEATURES
    }
    reference_values["gehalt"] = str(features["gehalt"].mode().iloc[0])
    out_of_fold_predictions = (out_of_fold_probabilities >= balanced_threshold).astype("int8")

    return ModelBundle(
        pipeline=final_pipeline,
        balanced_threshold=balanced_threshold,
        sensitive_threshold=sensitive_threshold,
        balanced_metrics=balanced_metrics,
        sensitive_metrics=sensitive_metrics,
        confusion=matrix,
        feature_importance=feature_importance,
        reference_values=reference_values,
        oof_probabilities=out_of_fold_probabilities,
        oof_predictions=out_of_fold_predictions,
        trained_rows=len(training_data),
        unique_profiles=int(groups.nunique()),
        churn_rate=float(target.mean()),
    )


def predict_scores(bundle: ModelBundle, features: pd.DataFrame) -> np.ndarray:
    """Return the model's churn score for one or more normalized rows."""
    return bundle.pipeline.predict_proba(features.loc[:, FEATURES])[:, 1]


def local_sensitivity(
    bundle: ModelBundle,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Measure one-feature-at-a-time score changes against training references."""
    current = features.iloc[[0]].loc[:, FEATURES]
    current_score = float(predict_scores(bundle, current)[0])
    rows: list[dict[str, Any]] = []

    for feature in FEATURES:
        reference_frame = current.copy()
        reference_frame[feature] = bundle.reference_values[feature]
        reference_score = float(predict_scores(bundle, reference_frame)[0])
        rows.append(
            {
                "feature": feature,
                "label": DISPLAY_NAMES[feature],
                "current_value": current.iloc[0][feature],
                "reference_value": bundle.reference_values[feature],
                "score_change": current_score - reference_score,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "score_change", key=lambda values: values.abs(), ascending=False
    )


def find_similar_profiles(
    training_data: pd.DataFrame,
    features: pd.DataFrame,
    *,
    neighbors: int = 250,
) -> dict[str, float | int]:
    """Summarize outcomes for the nearest profiles in normalized feature space."""
    candidate = training_data.loc[:, FEATURES]
    current = features.iloc[0]
    distance = pd.Series(0.0, index=candidate.index)

    for feature in (
        "zufriedenheitsgrad",
        "anzahl_projekte",
        "durchschnittliche_monatliche_arbeitszeit",
    ):
        scale = float(candidate[feature].std()) or 1.0
        distance += ((candidate[feature] - float(current[feature])) / scale) ** 2

    for feature in ("arbeitsunfall", "foerderung_letzte_5_jahre", "gehalt"):
        distance += (candidate[feature] != current[feature]).astype(float)

    count = min(max(int(neighbors), 1), len(training_data))
    nearest_indices = distance.nsmallest(count).index
    nearest_target = training_data.loc[nearest_indices, TARGET]
    return {
        "count": count,
        "churn_rate": float(nearest_target.mean()),
        "average_distance": float(np.sqrt(distance.loc[nearest_indices]).mean()),
    }


def what_if_analysis(
    bundle: ModelBundle,
    features: pd.DataFrame,
    feature: str,
    training_data: pd.DataFrame,
    *,
    points: int = 40,
) -> pd.DataFrame:
    """Calculate a one-feature sensitivity curve for the current profile."""
    if feature not in FEATURES:
        raise ValueError(f"Unbekanntes Feature: {feature}")

    if feature == "gehalt":
        values: list[Any] = ["low", "medium", "high"]
    elif feature in ("arbeitsunfall", "foerderung_letzte_5_jahre"):
        values = [0, 1]
    else:
        minimum = float(training_data[feature].min())
        maximum = float(training_data[feature].max())
        if feature in INTEGER_FEATURES:
            values = np.unique(np.rint(np.linspace(minimum, maximum, points))).astype(int).tolist()
        else:
            values = np.linspace(minimum, maximum, points).tolist()

    scenarios = pd.concat([features.iloc[[0]]] * len(values), ignore_index=True)
    # Direct assignment deliberately replaces the column dtype. This is needed for
    # continuous what-if grids when the current profile contains integer values.
    scenarios[feature] = values
    scores = predict_scores(bundle, scenarios)
    labels = [SALARY_DISPLAY.get(str(value), str(value)) for value in values]
    return pd.DataFrame({"value": values, "label": labels, "score": scores})
