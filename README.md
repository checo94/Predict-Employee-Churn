# Fluktuationsradar

[![Quality checks](https://github.com/kostasppz/Predict-Employee-Churn/actions/workflows/ci.yml/badge.svg)](https://github.com/kostasppz/Predict-Employee-Churn/actions/workflows/ci.yml)

Eine deutschsprachige Streamlit-Anwendung zur transparenten Analyse von
Mitarbeiterfluktuation. Die Anwendung unterstützt Einzelanalysen und CSV-/Excel-Dateien,
zeigt die geprüfte Modellgüte und macht Sensitivitäten sichtbar.

> **Wichtiger Verwendungshinweis:** Das Ergebnis ist ein statistischer Frühwarnindikator.
> Es ist keine Feststellung einer individuellen Kündigungsabsicht und darf nicht allein für
> Kündigungen, Beförderungen, Vergütung oder andere Personalentscheidungen verwendet werden.

## Funktionen

- Moderne, responsive Streamlit-Oberfläche in deutscher Sprache
- Einzelanalyse mit Abwanderungs- und Bleibensscore
- Zwei dokumentierte Betriebsmodi:
  - **Ausgewogen:** datenbasierte Schwelle mit F1-Optimierung
  - **Hohe Sensitivität:** F2-Optimierung für höheren Recall bei mehr Fehlalarmen
- What-if-Analyse für jeweils ein verändertes Merkmal
- Vergleich mit ähnlichen historischen Profilen
- Modellunabhängige Permutation Importance auf einer isolierten Prüffalte
- Vektorisierte Validierung von CSV- und Excel-Dateien
- Download der Stapelergebnisse als Excel-kompatible UTF-8-CSV
- Datenschutzfreundlicher Verlauf nur innerhalb der aktuellen Sitzung
- Automatische Qualitätsprüfungen mit Ruff und Pytest

## Modell und Evaluation

Die Pipeline verwendet einen `HistGradientBoostingClassifier`. Dieses Verfahren verarbeitet
nichtlineare Zusammenhänge effizient und benötigt deutlich weniger Modellobjekte als große
Random-Forest-Ensembles. Die Datenvorbereitung ist Bestandteil derselben Scikit-learn-Pipeline:

- Median-Imputation für numerische Werte
- Modus-Imputation und One-Hot-Encoding für die Gehaltsstufe
- feste Zufallsbasis für reproduzierbare Ergebnisse
- fünfteilige `StratifiedGroupKFold`-Kreuzvalidierung
- identische Merkmalsprofile immer in derselben Falte
- Out-of-Fold-Prognosen für alle angezeigten Qualitätskennzahlen
- F1- und F2-optimierte Schwellen ausschließlich aus Out-of-Fold-Ergebnissen

Die Gruppierung ist wichtig: Der Quelldatensatz enthält wiederholte Merkmalsprofile. Eine
gewöhnliche zufällige Aufteilung könnte dieselben Profile gleichzeitig in Training und Test
platzieren und dadurch eine zu optimistische Genauigkeit melden.

### Verifizierte Out-of-Fold-Ergebnisse

| Modus | Schwelle | Balanced Accuracy | Präzision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ausgewogen | 0,369 | 91,45 % | 85,83 % | 87,38 % | 0,866 |
| Hohe Sensitivität | 0,149 | 91,95 % | 78,18 % | 91,88 % | 0,845 |

Schwellenunabhängig erreicht das Modell `ROC-AUC = 0,966` und
`Average Precision = 0,935`. Die Werte werden bei jedem Training reproduzierbar aus
14.999 Out-of-Fold-Prognosen berechnet. Weitere Details stehen in der
[Modellkarte](MODEL_CARD.md).

## Verwendete Merkmale

| Spalte | Bedeutung | Werte |
| --- | --- | --- |
| `zufriedenheitsgrad` | Zufriedenheitswert | 0–100 |
| `anzahl_projekte` | Anzahl paralleler Projekte | im Trainingsdatensatz 2–7 |
| `durchschnittliche_monatliche_arbeitszeit` | Arbeitsstunden pro Monat | im Trainingsdatensatz 96–310 |
| `arbeitsunfall` | Arbeitsunfall | 0 oder 1 |
| `foerderung_letzte_5_jahre` | Förderung/Beförderung | 0 oder 1 |
| `gehalt` | Gehaltsstufe | low/medium/high, niedrig/mittel/hoch oder 1/2/3 |

Für Datei-Uploads werden außerdem die englischen Originalspalten unterstützt:
`satisfaction_level`, `number_project`, `average_montly_hours`, `Work_accident`,
`promotion_last_5years` und `salary`.

## Lokal starten

Voraussetzung ist Python 3.12.

### Windows PowerShell

```powershell
git clone https://github.com/kostasppz/Predict-Employee-Churn.git
cd Predict-Employee-Churn
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Linux oder macOS

```bash
git clone https://github.com/kostasppz/Predict-Employee-Churn.git
cd Predict-Employee-Churn
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit öffnet die Anwendung standardmäßig unter `http://localhost:8501`.

## Auf Streamlit Community Cloud bereitstellen

1. In Streamlit Community Cloud **Create app** auswählen.
2. Repository `kostasppz/Predict-Employee-Churn` verbinden.
3. Branch `main` auswählen.
4. Als Einstiegspunkt `app.py` eintragen.
5. Unter **Advanced settings** Python `3.12` auswählen.
6. **Deploy** starten.

Alle benötigten Python-Pakete stehen fest versioniert in `requirements.txt`. Die Anwendung
benötigt keine externen Datenbank- oder Plattformdienste. Ports und Containerisierung werden
von Streamlit Community Cloud verwaltet.

## Parallel auf SAP BTP bereitstellen

Das Repository enthält zusätzlich ein Cloud-Foundry-Manifest für SAP BTP. Beide Zielplattformen
verwenden dieselbe Anwendung und denselben Modellcode; die BTP-Konfiguration beeinflusst die
Bereitstellung auf Streamlit Community Cloud nicht.

Voraussetzungen sind ein SAP-BTP-Cloud-Foundry-Space und eine dort angemeldete CF CLI. Danach im
Stammverzeichnis des Repositorys ausführen:

```bash
cf target -o <ORG> -s <SPACE>
cf push
```

`manifest.yml` stellt die Anwendung unter dem Namen `employee-churn-app` bereit. Der Startbefehl
bindet Streamlit an alle Netzwerkschnittstellen und verwendet den von Cloud Foundry vergebenen
Port. `runtime.txt` hält BTP, CI und Streamlit Community Cloud auf Python 3.12. Die zufällige Route
verhindert Namenskonflikte in gemeinsam genutzten BTP-Domains.

Status, Route und letzte Protokolle lassen sich anschließend prüfen mit:

```bash
cf app employee-churn-app
cf logs employee-churn-app --recent
```

## Entwicklung und Qualitätssicherung

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q
python -m compileall -q app.py modeling.py
```

Bei jedem Push und Pull Request führt GitHub Actions diese Prüfungen mit Python 3.12 aus.

## Projektstruktur

```text
.
├── .github/workflows/ci.yml    # Continuous Integration
├── .streamlit/config.toml      # Theme und sichere App-Defaults
├── .cfignore                   # Ausschlüsse für den BTP-Upload
├── tests/                      # Daten-, Modell- und Validierungstests
├── HCM_Employee_Churn.csv      # intakter Quelldatensatz
├── MODEL_CARD.md               # Einsatzbereich, Messwerte und Grenzen
├── app.py                      # Streamlit-Oberfläche
├── manifest.yml                # SAP-BTP-Cloud-Foundry-Konfiguration
├── modeling.py                 # ML-, Evaluations- und Validierungslogik
├── pyproject.toml              # Ruff- und Pytest-Konfiguration
├── runtime.txt                 # Python-Laufzeit für den BTP-Buildpack
└── requirements.txt            # Community-Cloud-Abhängigkeiten
```

## Fachliche Grenzen

Das Modell kann nur Zusammenhänge aus den sechs vorhandenen Eingabemerkmalen lernen. Abteilung,
Betriebszugehörigkeit, Funktion, regionale Arbeitsmarktlage und viele weitere mögliche Faktoren
sind nicht im Datensatz enthalten. Eine seriöse Anwendung kann daher nicht „jedes Detail“ einer
realen Fluktuationsentscheidung erkennen.

Vor einem betrieblichen Einsatz sind mindestens folgende Schritte erforderlich:

- Datenherkunft, Rechtsgrundlage und Zweckbindung dokumentieren
- Arbeitnehmervertretung und Datenschutzbeauftragte einbeziehen
- Leistung getrennt nach relevanten Gruppen auf Fairness prüfen
- Daten- und Konzeptdrift regelmäßig überwachen
- fachliche Prüfung und Einspruchsmöglichkeit sicherstellen
- ausschließlich unterstützende, positive Bindungsmaßnahmen ableiten
