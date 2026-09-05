"""Build the validated production model artifact outside the web process."""

from pathlib import Path

from modeling import load_training_data, save_model_artifact, train_model

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "HCM_Employee_Churn.csv"
ARTIFACT_PATH = ROOT / "artifacts" / "churn_model.joblib"


def main() -> None:
    training_data = load_training_data(DATA_PATH)
    model = train_model(training_data)
    save_model_artifact(model, ARTIFACT_PATH, DATA_PATH)
    print(
        f"Saved model {model.model_version} to {ARTIFACT_PATH.relative_to(ROOT)} "
        f"with ROC-AUC {model.balanced_metrics['roc_auc']:.3f}."
    )


if __name__ == "__main__":
    main()
