from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cloud_foundry_manifest_keeps_streamlit_portable() -> None:
    manifest = (ROOT / "manifest.yml").read_text(encoding="utf-8")

    assert "name: employee-churn-app" in manifest
    assert "python_buildpack" in manifest
    assert "exec streamlit run app.py" in manifest
    assert "--server.address=0.0.0.0" in manifest
    assert "--server.port=$PORT" in manifest
    assert "health-check-type: port" in manifest


def test_cloud_foundry_runtime_matches_ci_and_documentation() -> None:
    assert (ROOT / "runtime.txt").read_text(encoding="utf-8").strip() == "python-3.12.x"
