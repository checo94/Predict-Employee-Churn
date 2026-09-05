"""Modern Streamlit interface for responsible employee churn exploration."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from modeling import (
    DISPLAY_NAMES,
    FEATURES,
    MODEL_VERSION,
    SALARY_DISPLAY,
    DataValidationError,
    ModelBundle,
    find_similar_profiles,
    load_model_artifact,
    load_training_data,
    local_sensitivity,
    predict_scores,
    validate_feature_frame,
    what_if_analysis,
)

APP_ROOT = Path(__file__).resolve().parent
DATA_PATH = APP_ROOT / "HCM_Employee_Churn.csv"
MODEL_PATH = APP_ROOT / "artifacts" / "churn_model.joblib"

st.set_page_config(
    page_title="Fluktuationsradar | People Analytics",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    """Add a restrained visual layer while keeping native Streamlit accessibility."""
    st.markdown(
        """
        <style>
            :root {
                --ink: #132238;
                --muted: #5d6b7c;
                --teal: #087f78;
                --teal-soft: #e7f6f3;
                --amber: #b86b00;
                --amber-soft: #fff4dc;
                --surface: #ffffff;
                --line: #dbe4ec;
            }
            .stApp { background: #f6f8fb; }
            .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
            [data-testid="stSidebar"] { border-right: 1px solid var(--line); }
            [data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 0.9rem 1rem;
                box-shadow: 0 5px 18px rgba(19, 34, 56, 0.04);
            }
            .hero {
                padding: 1.8rem 2rem;
                border-radius: 22px;
                color: white;
                background:
                    radial-gradient(circle at 90% 10%, rgba(255,255,255,.18), transparent 30%),
                    linear-gradient(120deg, #12304a 0%, #087f78 100%);
                box-shadow: 0 14px 34px rgba(18, 48, 74, .18);
                margin-bottom: 1.25rem;
            }
            .hero-kicker {
                text-transform: uppercase;
                letter-spacing: .12em;
                font-size: .75rem;
                font-weight: 700;
                opacity: .82;
                margin-bottom: .4rem;
            }
            .hero h1 { color: white; font-size: 2.35rem; margin: 0 0 .45rem 0; }
            .hero p { margin: 0; max-width: 780px; font-size: 1.02rem; opacity: .9; }
            .result-card {
                border-radius: 18px;
                padding: 1.2rem 1.35rem;
                margin: .3rem 0 1rem 0;
                border: 1px solid var(--line);
                background: var(--surface);
            }
            .result-card.low { border-left: 6px solid var(--teal); background: var(--teal-soft); }
            .result-card.high { border-left: 6px solid var(--amber); background: var(--amber-soft); }
            .result-label { color: var(--muted); font-size: .82rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
            .result-title { color: var(--ink); font-size: 1.35rem; font-weight: 750; margin-top: .2rem; }
            .result-copy { color: var(--muted); margin-top: .35rem; }
            .pill {
                display: inline-block;
                padding: .28rem .62rem;
                border-radius: 999px;
                background: rgba(255,255,255,.82);
                border: 1px solid rgba(8,127,120,.22);
                color: var(--teal);
                font-size: .78rem;
                font-weight: 700;
                margin-right: .35rem;
                margin-top: .65rem;
            }
            .method-note {
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 1rem 1.1rem;
                background: white;
                color: var(--muted);
            }
            .footer {
                color: var(--muted);
                text-align: center;
                font-size: .82rem;
                padding-top: 1rem;
            }
            div[data-testid="stForm"] {
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 1rem;
                background: rgba(255,255,255,.74);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_training_data(path: str, modified_time: int) -> pd.DataFrame:
    """Cache validated source data and invalidate when the file changes."""
    del modified_time
    return load_training_data(path)


@st.cache_resource(show_spinner=False)
def get_model_bundle(
    data_path: str,
    data_modified_time: int,
    model_path: str,
    model_modified_time: int,
) -> ModelBundle:
    """Load the validated production model once per artifact version."""
    del data_modified_time, model_modified_time
    return load_model_artifact(model_path, data_path)


def percent(value: float) -> str:
    return f"{value * 100:.1f} %"


def read_uploaded_table(uploaded_file: object) -> pd.DataFrame:
    """Read CSVs with common German encodings/delimiters or an XLSX workbook."""
    file_name = str(getattr(uploaded_file, "name", "")).lower()
    payload = uploaded_file.getvalue()
    if file_name.endswith(".xlsx"):
        return pd.read_excel(BytesIO(payload))

    decoding_errors: list[UnicodeDecodeError] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return pd.read_csv(
                BytesIO(payload),
                sep=None,
                engine="python",
                encoding=encoding,
            )
        except UnicodeDecodeError as error:
            decoding_errors.append(error)
    raise decoding_errors[-1]


def safe_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Create an Excel-friendly CSV while neutralizing spreadsheet formulas."""
    safe_frame = frame.copy()
    text_columns = safe_frame.select_dtypes(include=("object", "string")).columns
    dangerous_prefixes = ("=", "+", "-", "@", "\t", "\r")
    for column in text_columns:
        safe_frame[column] = safe_frame[column].map(
            lambda value: (
                "'" + value
                if isinstance(value, str) and value.startswith(dangerous_prefixes)
                else value
            )
        )
    return safe_frame.to_csv(index=False).encode("utf-8-sig")


def format_feature_value(feature: str, value: object) -> str:
    if feature == "gehalt":
        return SALARY_DISPLAY.get(str(value), str(value))
    if feature in ("arbeitsunfall", "foerderung_letzte_5_jahre"):
        return "Ja" if int(float(value)) == 1 else "Nein"
    if feature == "zufriedenheitsgrad":
        return f"{float(value):.0f} %"
    if feature == "durchschnittliche_monatliche_arbeitszeit":
        return f"{float(value):.0f} Std."
    return f"{float(value):.0f}"


def create_importance_figure(importance: pd.DataFrame) -> plt.Figure:
    ordered = importance.sort_values("importance", ascending=True)
    values = ordered["importance"].clip(lower=0)
    colors = ["#98bfc0" if value < values.max() * 0.5 else "#087f78" for value in values]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.barh(ordered["label"], values, color=colors, height=0.62)
    axis.set_xlabel("Rückgang der Average Precision bei Permutation")
    axis.set_ylabel("")
    axis.grid(axis="x", alpha=0.18)
    axis.spines[["top", "right", "left"]].set_visible(False)
    figure.tight_layout()
    return figure


def create_sensitivity_figure(sensitivity: pd.DataFrame) -> plt.Figure:
    ordered = sensitivity.sort_values("score_change", ascending=True)
    colors = np.where(ordered["score_change"] >= 0, "#b86b00", "#087f78")
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.barh(ordered["label"], ordered["score_change"] * 100, color=colors, height=0.62)
    axis.axvline(0, color="#132238", linewidth=1)
    axis.set_xlabel("Änderung des Abwanderungsscores in Prozentpunkten")
    axis.set_ylabel("")
    axis.grid(axis="x", alpha=0.18)
    axis.spines[["top", "right", "left"]].set_visible(False)
    figure.tight_layout()
    return figure


def create_confusion_figure(matrix: np.ndarray) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(5.8, 4.4))
    image = axis.imshow(matrix, cmap="BuGn")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{int(matrix[row, column]):,}".replace(",", "."),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="white" if matrix[row, column] > matrix.max() * 0.55 else "#132238",
            )
    axis.set_xticks([0, 1], ["Bleibt", "Signal"])
    axis.set_yticks([0, 1], ["Bleibt", "Geht"])
    axis.set_xlabel("Modellklassifikation")
    axis.set_ylabel("Tatsächlicher Ausgang")
    axis.set_title("Out-of-Fold-Konfusionsmatrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure


def create_what_if_figure(
    analysis: pd.DataFrame,
    feature: str,
    current_value: object,
    threshold: float,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9, 4.6))
    if feature in ("gehalt", "arbeitsunfall", "foerderung_letzte_5_jahre"):
        labels = analysis["label"].replace({"0": "Nein", "1": "Ja"})
        axis.bar(labels, analysis["score"] * 100, color="#087f78", width=0.58)
    else:
        axis.plot(analysis["value"], analysis["score"] * 100, color="#087f78", linewidth=2.8)
        axis.fill_between(
            analysis["value"].astype(float),
            analysis["score"] * 100,
            color="#087f78",
            alpha=0.12,
        )
        axis.axvline(float(current_value), color="#132238", linestyle=":", label="Aktueller Wert")
    axis.axhline(threshold * 100, color="#b86b00", linestyle="--", label="Signalschwelle")
    axis.set_ylabel("Abwanderungsscore (%)")
    axis.set_xlabel(DISPLAY_NAMES[feature])
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


def create_distribution_figure(training_data: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    axes = axes.ravel()
    stayed = training_data[training_data["left"] == 0]
    left = training_data[training_data["left"] == 1]

    for axis, feature in zip(axes, FEATURES, strict=True):
        if feature in ("arbeitsunfall", "foerderung_letzte_5_jahre", "gehalt"):
            counts = training_data.groupby([feature, "left"]).size().unstack(fill_value=0)
            counts.plot(kind="bar", ax=axis, color=["#61a8a2", "#d59a44"], width=0.72)
            axis.tick_params(axis="x", rotation=0)
            axis.legend(["Bleibt", "Geht"], frameon=False, fontsize=8)
        else:
            axis.hist(
                [stayed[feature], left[feature]],
                bins=24,
                label=["Bleibt", "Geht"],
                color=["#61a8a2", "#d59a44"],
                alpha=0.72,
            )
            axis.legend(frameon=False, fontsize=8)
        axis.set_title(DISPLAY_NAMES[feature], fontsize=10, fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("Anzahl")
        axis.grid(axis="y", alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
    return figure


def render_validation_messages(errors: tuple[str, ...], warnings: tuple[str, ...]) -> None:
    if errors:
        st.error("Die Datei kann noch nicht ausgewertet werden.")
        for error in errors:
            st.markdown(f"- {error}")
    if warnings:
        with st.expander(f"{len(warnings)} Datenhinweis(e)"):
            for warning in warnings:
                st.markdown(f"- {warning}")


apply_theme()

try:
    data_modified_time = DATA_PATH.stat().st_mtime_ns
    model_modified_time = MODEL_PATH.stat().st_mtime_ns
    with st.spinner("Validiertes Modell wird geladen …"):
        data = get_training_data(str(DATA_PATH), data_modified_time)
        model = get_model_bundle(
            str(DATA_PATH),
            data_modified_time,
            str(MODEL_PATH),
            model_modified_time,
        )
except (OSError, DataValidationError, ValueError) as error:
    st.error(f"Die Anwendung konnte nicht gestartet werden: {error}")
    st.stop()

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">People Analytics · Responsible ML</div>
        <h1>Fluktuationsradar</h1>
        <p>
            Ein transparenter Frühwarnindikator für Mitarbeiterfluktuation – mit
            Einzelanalyse, Stapelverarbeitung und nachvollziehbarer Modellqualität.
        </p>
        <span class="pill">Datenschutzfreundliche Sitzung</span>
        <span class="pill">Leckagearme Evaluation</span>
        <span class="pill">Keine automatisierte Personalentscheidung</span>
    </section>
    """,
    unsafe_allow_html=True,
)

headline_columns = st.columns(4)
headline_columns[0].metric("Trainingsdatensätze", f"{model.trained_rows:,}".replace(",", "."))
headline_columns[1].metric("Eindeutige Profile", f"{model.unique_profiles:,}".replace(",", "."))
headline_columns[2].metric("ROC-AUC", f"{model.balanced_metrics['roc_auc']:.3f}")
headline_columns[3].metric("Recall · ausgewogen", percent(model.balanced_metrics["recall"]))

with st.sidebar:
    st.markdown("## Analyse konfigurieren")
    analysis_mode = st.radio(
        "Betriebsmodus",
        options=("Ausgewogen", "Hohe Sensitivität"),
        help=(
            "Der ausgewogene Modus optimiert F1. Hohe Sensitivität gewichtet das "
            "Erkennen möglicher Abwanderungen stärker und erzeugt mehr Fehlalarme."
        ),
    )
    active_threshold = (
        model.balanced_threshold if analysis_mode == "Ausgewogen" else model.sensitive_threshold
    )
    active_metrics = (
        model.balanced_metrics if analysis_mode == "Ausgewogen" else model.sensitive_metrics
    )
    st.caption(
        f"Signalschwelle {active_threshold:.2f} · Recall {percent(active_metrics['recall'])} · "
        f"Präzision {percent(active_metrics['precision'])}"
    )

    defaults = model.reference_values
    with st.form("employee_profile", border=False):
        st.markdown("### Mitarbeiterprofil")
        satisfaction = st.slider(
            "Zufriedenheitsgrad",
            min_value=0,
            max_value=100,
            value=int(round(defaults["zufriedenheitsgrad"])),
            help="Selbsteinschätzung oder standardisierter Befragungswert in Prozent.",
        )
        projects = st.slider(
            "Anzahl Projekte",
            min_value=int(data["anzahl_projekte"].min()),
            max_value=int(data["anzahl_projekte"].max()),
            value=int(round(defaults["anzahl_projekte"])),
        )
        monthly_hours = st.slider(
            "Monatliche Arbeitszeit",
            min_value=int(data["durchschnittliche_monatliche_arbeitszeit"].min()),
            max_value=int(data["durchschnittliche_monatliche_arbeitszeit"].max()),
            value=int(round(defaults["durchschnittliche_monatliche_arbeitszeit"])),
            step=1,
        )
        work_accident = st.selectbox("Arbeitsunfall", options=("Nein", "Ja"))
        promoted = st.selectbox("Förderung in den letzten 5 Jahren", options=("Nein", "Ja"))
        salary_label = st.selectbox(
            "Gehaltsstufe",
            options=("Niedrig", "Mittel", "Hoch"),
            index=("Niedrig", "Mittel", "Hoch").index(SALARY_DISPLAY[str(defaults["gehalt"])]),
        )
        submitted = st.form_submit_button(
            "Profil analysieren",
            type="primary",
            width="stretch",
        )

    st.info(
        "Das Ergebnis unterstützt eine vertiefende Analyse. Es darf nicht allein für "
        "Kündigungen, Beförderungen oder andere Personalmaßnahmen verwendet werden."
    )

salary_reverse = {label: value for value, label in SALARY_DISPLAY.items()}
profile = pd.DataFrame(
    [
        {
            "zufriedenheitsgrad": satisfaction,
            "anzahl_projekte": projects,
            "durchschnittliche_monatliche_arbeitszeit": monthly_hours,
            "arbeitsunfall": int(work_accident == "Ja"),
            "foerderung_letzte_5_jahre": int(promoted == "Ja"),
            "gehalt": salary_reverse[salary_label],
        }
    ]
)
churn_score = float(predict_scores(model, profile)[0])
has_signal = churn_score >= active_threshold

if submitted:
    st.session_state.prediction_history.insert(
        0,
        {
            "Zeitpunkt": datetime.now().astimezone().strftime("%d.%m.%Y %H:%M:%S"),
            "Zufriedenheit": satisfaction,
            "Projekte": projects,
            "Arbeitsstunden": monthly_hours,
            "Arbeitsunfall": work_accident,
            "Förderung": promoted,
            "Gehalt": salary_label,
            "Abwanderungsscore (%)": round(churn_score * 100, 1),
            "Modellsignal": "Erhöht" if has_signal else "Unauffällig",
            "Modus": analysis_mode,
        },
    )
    st.session_state.prediction_history = st.session_state.prediction_history[:100]

single_tab, batch_tab, quality_tab, method_tab, history_tab = st.tabs(
    [
        "Einzelanalyse",
        "Stapelprognose",
        "Modellqualität",
        "Daten & Methodik",
        "Sitzungsverlauf",
    ]
)

with single_tab:
    result_class = "high" if has_signal else "low"
    result_title = "Erhöhtes Abwanderungssignal" if has_signal else "Unauffälliges Modellsignal"
    result_copy = (
        "Das Profil liegt oberhalb der gewählten Frühwarnschwelle. Prüfen Sie den Kontext "
        "in einem fairen, menschlich geführten Gespräch."
        if has_signal
        else "Das Profil liegt unterhalb der gewählten Frühwarnschwelle. Das schließt eine "
        "spätere Abwanderung nicht aus."
    )
    st.markdown(
        f"""
        <div class="result-card {result_class}">
            <div class="result-label">Aktuelles Ergebnis · {analysis_mode}</div>
            <div class="result-title">{result_title}</div>
            <div class="result-copy">{result_copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result_columns = st.columns(4)
    result_columns[0].metric("Abwanderungsscore", percent(churn_score))
    result_columns[1].metric("Bleibensscore", percent(1 - churn_score))
    result_columns[2].metric("Signalschwelle", f"{active_threshold:.2f}")
    result_columns[3].metric("Betriebsmodus", analysis_mode)
    st.progress(
        churn_score,
        text=f"Modellscore: {percent(churn_score)} · Schwelle: {percent(active_threshold)}",
    )
    st.caption(
        "Der Score ist eine Modellschätzung aus sechs Merkmalen und keine Gewissheit oder "
        "Kausalaussage."
    )

    insight_column, peer_column = st.columns([1.65, 1])
    with insight_column:
        st.subheader("Einfluss des aktuellen Profils")
        sensitivity = local_sensitivity(model, profile)
        sensitivity_figure = create_sensitivity_figure(sensitivity)
        st.pyplot(sensitivity_figure, width="stretch")
        plt.close(sensitivity_figure)
        st.caption(
            "Jeder Balken zeigt die Scoreänderung, wenn nur dieses Merkmal auf den "
            "Trainingsreferenzwert gesetzt wird. Wechselwirkungen bleiben bestehen."
        )

    with peer_column:
        st.subheader("Ähnliche Profile")
        peer_summary = find_similar_profiles(data, profile)
        with st.container(border=True):
            st.metric(
                "Historische Fluktuationsrate",
                percent(float(peer_summary["churn_rate"])),
            )
            st.write(
                f"Aus den **{int(peer_summary['count'])} ähnlichsten Profilen** im "
                "Trainingsdatensatz."
            )
            st.caption(
                "Die Ähnlichkeit basiert auf standardisierten Merkmalsabständen. Sie ist "
                "deskriptiv und keine individuelle Erklärung."
            )

        detail_table = sensitivity.copy()
        detail_table["Aktueller Wert"] = [
            format_feature_value(feature, value)
            for feature, value in zip(
                detail_table["feature"], detail_table["current_value"], strict=True
            )
        ]
        detail_table["Referenz"] = [
            format_feature_value(feature, value)
            for feature, value in zip(
                detail_table["feature"], detail_table["reference_value"], strict=True
            )
        ]
        detail_table["Scoreänderung"] = detail_table["score_change"].map(
            lambda value: f"{value * 100:+.1f} Pp."
        )
        st.dataframe(
            detail_table[["label", "Aktueller Wert", "Referenz", "Scoreänderung"]].rename(
                columns={"label": "Merkmal"}
            ),
            hide_index=True,
            width="stretch",
        )

    st.divider()
    st.subheader("What-if-Analyse")
    st.write(
        "Verändern Sie genau ein Merkmal. Die Darstellung zeigt Sensitivität, keine "
        "Handlungsempfehlung und keinen kausalen Effekt."
    )
    what_if_feature = st.selectbox(
        "Merkmal auswählen",
        options=FEATURES,
        format_func=lambda feature: DISPLAY_NAMES[feature],
        key="what_if_feature",
    )
    what_if_data = what_if_analysis(model, profile, what_if_feature, data)
    what_if_figure = create_what_if_figure(
        what_if_data,
        what_if_feature,
        profile.iloc[0][what_if_feature],
        active_threshold,
    )
    st.pyplot(what_if_figure, width="stretch")
    plt.close(what_if_figure)

with batch_tab:
    st.subheader("Mehrere Profile sicher auswerten")
    st.write(
        "CSV- und Excel-Dateien werden vektorisiert geprüft. Deutsche und englische "
        "Spaltennamen sowie deutsche, englische und numerische Gehaltsstufen werden erkannt."
    )
    template = pd.DataFrame(
        [
            {
                "zufriedenheitsgrad": 68,
                "anzahl_projekte": 4,
                "durchschnittliche_monatliche_arbeitszeit": 190,
                "arbeitsunfall": 0,
                "foerderung_letzte_5_jahre": 0,
                "gehalt": "mittel",
            },
            {
                "zufriedenheitsgrad": 32,
                "anzahl_projekte": 6,
                "durchschnittliche_monatliche_arbeitszeit": 265,
                "arbeitsunfall": 0,
                "foerderung_letzte_5_jahre": 0,
                "gehalt": "niedrig",
            },
        ]
    )
    st.download_button(
        "CSV-Vorlage herunterladen",
        data=safe_csv_bytes(template),
        file_name="vorlage_mitarbeiterprofile.csv",
        mime="text/csv",
    )
    uploaded_file = st.file_uploader(
        "CSV- oder Excel-Datei auswählen",
        type=("csv", "xlsx"),
        help="Maximal 15 MB und 50.000 Datenzeilen.",
    )

    if uploaded_file is not None:
        if uploaded_file.size > 15 * 1024 * 1024:
            st.error("Die Datei ist größer als 15 MB.")
        else:
            try:
                uploaded_data = read_uploaded_table(uploaded_file)
            except (ValueError, OSError, UnicodeError) as error:
                st.error(f"Die Datei konnte nicht gelesen werden: {error}")
            else:
                uploaded_data = uploaded_data.dropna(how="all").reset_index(drop=True)
                if len(uploaded_data) > 50_000:
                    st.error("Die Datei enthält mehr als 50.000 Datenzeilen.")
                else:
                    report = validate_feature_frame(uploaded_data, allow_missing=True)
                    render_validation_messages(report.errors, report.warnings)
                    if not report.errors:
                        batch_scores = predict_scores(model, report.features)
                        batch_signals = batch_scores >= active_threshold
                        output = uploaded_data.copy()
                        output["Abwanderungsscore (%)"] = np.round(batch_scores * 100, 1)
                        output["Bleibensscore (%)"] = np.round((1 - batch_scores) * 100, 1)
                        output["Modellsignal"] = np.where(batch_signals, "Erhöht", "Unauffällig")
                        output["Analysemodus"] = analysis_mode

                        summary_columns = st.columns(3)
                        summary_columns[0].metric("Verarbeitete Profile", len(output))
                        summary_columns[1].metric("Erhöhte Signale", int(batch_signals.sum()))
                        summary_columns[2].metric(
                            "Mittlerer Score", percent(float(batch_scores.mean()))
                        )
                        st.download_button(
                            "Auswertung als CSV herunterladen",
                            data=safe_csv_bytes(output),
                            file_name="fluktuationsradar_auswertung.csv",
                            mime="text/csv",
                            type="primary",
                        )
                        st.dataframe(
                            output.head(2_000),
                            hide_index=True,
                            width="stretch",
                        )
                        if len(output) > 2_000:
                            st.caption(
                                "In der Vorschau werden 2.000 Zeilen angezeigt; der Download "
                                "enthält die vollständige Auswertung."
                            )

with quality_tab:
    st.subheader("Nachvollziehbare Modellgüte")
    st.markdown(
        """
        <div class="method-note">
            Die Kennzahlen stammen aus fünffacher, stratifizierter Gruppen-Kreuzvalidierung.
            Identische Merkmalsprofile werden derselben Falte zugeordnet und können dadurch
            nicht gleichzeitig in Training und Prüfung vorkommen. Das reduziert die sonst
            deutlich zu optimistische Bewertung durch Duplikat-Leckage.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(5)
    metric_columns[0].metric("Balanced Accuracy", percent(active_metrics["balanced_accuracy"]))
    metric_columns[1].metric("Recall", percent(active_metrics["recall"]))
    metric_columns[2].metric("Präzision", percent(active_metrics["precision"]))
    metric_columns[3].metric("F1", f"{active_metrics['f1']:.3f}")
    metric_columns[4].metric("Average Precision", f"{active_metrics['average_precision']:.3f}")

    mode_comparison = pd.DataFrame(
        [
            {
                "Modus": "Ausgewogen",
                "Schwelle": model.balanced_threshold,
                "Recall": model.balanced_metrics["recall"],
                "Präzision": model.balanced_metrics["precision"],
                "F1": model.balanced_metrics["f1"],
                "Spezifität": model.balanced_metrics["specificity"],
            },
            {
                "Modus": "Hohe Sensitivität",
                "Schwelle": model.sensitive_threshold,
                "Recall": model.sensitive_metrics["recall"],
                "Präzision": model.sensitive_metrics["precision"],
                "F1": model.sensitive_metrics["f1"],
                "Spezifität": model.sensitive_metrics["specificity"],
            },
        ]
    )
    for column in ("Recall", "Präzision", "Spezifität"):
        mode_comparison[column] = mode_comparison[column].map(percent)
    mode_comparison["Schwelle"] = mode_comparison["Schwelle"].map(lambda value: f"{value:.3f}")
    mode_comparison["F1"] = mode_comparison["F1"].map(lambda value: f"{value:.3f}")

    st.dataframe(mode_comparison, hide_index=True, width="stretch")

    confusion_column, importance_column = st.columns([1, 1.5])
    with confusion_column:
        st.markdown("#### Fehlerbild · ausgewogener Modus")
        confusion_figure = create_confusion_figure(model.confusion)
        st.pyplot(confusion_figure, width="stretch")
        plt.close(confusion_figure)
    with importance_column:
        st.markdown("#### Globale Merkmalswichtigkeit")
        importance_figure = create_importance_figure(model.feature_importance)
        st.pyplot(importance_figure, width="stretch")
        plt.close(importance_figure)
        st.caption(
            "Permutation Importance misst den Leistungsabfall auf einer isolierten Prüffalte. "
            "Sie beschreibt Relevanz im Modell, nicht Ursache oder Fairness."
        )

with method_tab:
    st.subheader("Datenbasis und Modellmethodik")
    distribution_figure = create_distribution_figure(data)
    st.pyplot(distribution_figure, width="stretch")
    plt.close(distribution_figure)

    dictionary = pd.DataFrame(
        [
            ("Zufriedenheitsgrad", "Numerisch", "0–100 %"),
            ("Anzahl Projekte", "Ganzzahl", "Im Datensatz 2–7"),
            ("Monatliche Arbeitszeit", "Ganzzahl", "Im Datensatz 96–310 Stunden"),
            ("Arbeitsunfall", "Binär", "Nein / Ja"),
            ("Förderung in den letzten 5 Jahren", "Binär", "Nein / Ja"),
            ("Gehaltsstufe", "Kategorie", "Niedrig / Mittel / Hoch"),
            ("Fluktuation", "Zielvariable", "Bleibt / Geht"),
        ],
        columns=("Merkmal", "Typ", "Wertebereich"),
    )
    st.dataframe(dictionary, hide_index=True, width="stretch")

    limitation_column, engineering_column = st.columns(2)
    with limitation_column:
        st.markdown("#### Fachliche Grenzen")
        st.markdown(
            """
            - Das Modell kennt nur die sechs dokumentierten Merkmale.
            - Abteilung, Betriebszugehörigkeit, Rolle und externe Arbeitsmarktfaktoren fehlen.
            - Historische Muster können Verzerrungen enthalten und ändern sich über die Zeit.
            - Scores dürfen nicht als Beweis für eine individuelle Kündigungsabsicht gelten.
            - Vor einem produktiven Einsatz sind Datenschutz-, Mitbestimmungs- und Fairnessprüfungen nötig.
            """
        )
    with engineering_column:
        st.markdown("#### Technische Verbesserungen")
        st.markdown(
            """
            - Intakter Quelldatensatz statt der beschädigten abgeleiteten CSV
            - Gecachtes Training statt Neutraining bei jeder Interaktion
            - Histogram Gradient Boosting für nichtlineare Zusammenhänge
            - One-Hot-Encoding statt künstlicher Abstände zwischen Gehaltsstufen
            - Gruppenbasierte Kreuzvalidierung gegen Duplikat-Leckage
            - F1- und F2-optimierte, datenbasierte Signalschwellen
            - Vektorisierte Datei- und Wertevalidierung
            """
        )

with history_tab:
    st.subheader("Analysen dieser Sitzung")
    st.write(
        "Aus Datenschutzgründen wird der Verlauf nur in der aktuellen Streamlit-Sitzung "
        "gehalten. Es werden keine eingegebenen Mitarbeiterprofile an einen externen Dienst gesendet."
    )
    if st.session_state.prediction_history:
        history = pd.DataFrame(st.session_state.prediction_history)
        st.download_button(
            "Sitzungsverlauf als CSV herunterladen",
            data=safe_csv_bytes(history),
            file_name="fluktuationsradar_sitzungsverlauf.csv",
            mime="text/csv",
        )
        if st.button("Sitzungsverlauf löschen"):
            st.session_state.prediction_history = []
            st.rerun()
        st.dataframe(history, hide_index=True, width="stretch")
    else:
        st.info(
            "Noch keine gespeicherte Analyse. Klicken Sie links auf „Profil analysieren“, "
            "um das aktuelle Profil in den Sitzungsverlauf aufzunehmen."
        )

st.markdown(
    f"""
    <div class="footer">
        Fluktuationsradar · Modell {MODEL_VERSION} · Streamlit Community Cloud ready<br>
        Frühwarnindikator für analytische und wissenschaftliche Zwecke – menschliche Prüfung erforderlich.
    </div>
    """,
    unsafe_allow_html=True,
)
