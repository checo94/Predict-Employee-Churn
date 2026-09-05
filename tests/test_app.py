from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_exceptions() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=120).run()

    assert not app.exception
    assert any("Fluktuationsradar" in element.value for element in app.markdown)
