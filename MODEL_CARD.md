# Modellkarte: Fluktuationsradar 2.1

## Modellzweck

Das Modell ist ein transparenter Frühwarnindikator für explorative People-Analytics-Analysen.
Es schätzt anhand sechs betrieblicher Merkmale, ob ein historisches Profil eher dem Muster
„bleibt“ oder „geht“ ähnelt. Der Score soll vertiefende, freiwillige und fair geführte
Bindungsgespräche unterstützen.

## Nicht vorgesehene Verwendung

Das Modell ist nicht für automatisierte oder allein maßgebliche Entscheidungen über Kündigung,
Beförderung, Vergütung, Einstellung, Leistungsbewertung oder Disziplinarmaßnahmen vorgesehen.
Der Score ist weder eine Diagnose noch ein Beweis für eine individuelle Kündigungsabsicht.

## Daten

- Datei: `HCM_Employee_Churn.csv`
- Beobachtungen: 14.999
- Eindeutige Merkmalsprofile: 11.119
- Positive Zielklasse „geht“: 3.557 (23,71 %)
- Eingaben: Zufriedenheit, Projekte, monatliche Arbeitszeit, Arbeitsunfall,
  Förderung in den letzten fünf Jahren und Gehaltsstufe
- Personenbezogene Identifikatoren und geschützte Merkmale sind nicht enthalten

Die Datenherkunft und ihre Repräsentativität müssen vor einem realen betrieblichen Einsatz
organisatorisch und rechtlich verifiziert werden.

## Verfahren

- Histogram Gradient Boosting mit festem Zufallswert
- Median-Imputation für numerische Eingaben
- Modus-Imputation und One-Hot-Encoding für Gehaltsstufen
- fünfteilige stratifizierte Gruppen-Kreuzvalidierung
- Gruppenschlüssel aus dem vollständigen Merkmalsprofil gegen Duplikat-Leckage
- globale Permutation Importance auf einer getrennten Prüffalte
- lokale Ein-Merkmal-Sensitivität ohne kausale oder additive Behauptung

## Training und Bereitstellung

Die vollständige Kreuzvalidierung, Schwellenoptimierung, Permutation Importance und das finale
Training werden außerhalb des Webprozesses ausgeführt. Das resultierende, versionierte
Produktionsartefakt wird zusammen mit dem Quellcode bereitgestellt. Beim Laden prüft die
Anwendung den SHA-256-Fingerabdruck der Trainingsdatei sowie die verwendeten Versionen von
Scikit-learn, NumPy, pandas und Joblib. Es werden ausschließlich Artefakte aus dem kontrollierten
Repository geladen; hochgeladene Modelldateien werden nicht deserialisiert.

Dadurch führt weder eine neue Streamlit-Sitzung noch eine zusätzliche Cloud-Foundry-Instanz ein
erneutes Training aus. Eine fachlich beabsichtigte Daten- oder Modelländerung erfordert dagegen
explizit eine neue Artefakterstellung und die vollständige automatisierte Prüfung.

## Evaluation

Alle Kennzahlen basieren auf Out-of-Fold-Prognosen. Die angezeigte Schwelle wird auf diesen
Prognosen optimiert.

| Kennzahl | Ausgewogen | Hohe Sensitivität |
| --- | ---: | ---: |
| Schwelle | 0,369 | 0,149 |
| Accuracy | 93,59 % | 91,99 % |
| Balanced Accuracy | 91,45 % | 91,95 % |
| Präzision | 85,83 % | 78,18 % |
| Recall | 87,38 % | 91,88 % |
| Spezifität | 95,52 % | 92,03 % |
| F1 | 0,866 | 0,845 |

Schwellenunabhängige Kennzahlen:

- ROC-AUC: 0,966
- Average Precision: 0,935
- Brier Score: 0,0476

Konfusionsmatrix des ausgewogenen Modus (`tatsächlich × prognostiziert`):

|  | Bleibt | Signal |
| --- | ---: | ---: |
| Bleibt | 10.929 | 513 |
| Geht | 449 | 3.108 |

## Wesentliche Grenzen

- Das Modell kennt nur sechs Merkmale und kann nicht alle Ursachen realer Fluktuation erfassen.
- Abteilung, Rolle, Betriebszugehörigkeit, Führung, Teamklima und Arbeitsmarkt fehlen.
- Es liegen keine geschützten Merkmale vor; gruppenbezogene Fairness kann daher nicht gemessen werden.
- Historische Zusammenhänge sind nicht automatisch kausal, fair oder zukünftig stabil.
- Werte außerhalb des Trainingsbereichs sind Extrapolationen und werden in der App markiert.
- Die Schwellenoptimierung auf denselben Out-of-Fold-Scores kann eine geringe Auswahloptimistik enthalten.

## Erforderliche Kontrollen für einen betrieblichen Einsatz

1. Zweck, Rechtsgrundlage, Datenminimierung und Löschkonzept festlegen.
2. Datenschutz, Arbeitnehmervertretung und Informationssicherheit einbeziehen.
3. Repräsentative Organisationsdaten mit dokumentierter Qualität verwenden.
4. Fairness, Kalibrierung, Fehlerraten und Drift regelmäßig prüfen.
5. Jede Einzelfallbewertung durch qualifizierte Menschen prüfen lassen.
6. Betroffenen Transparenz, Korrekturmöglichkeiten und einen Einspruchsweg geben.
