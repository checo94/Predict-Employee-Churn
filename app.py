import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set page configuration
st.set_page_config(
    page_title="Mitarbeiterabwanderung Vorhersagen",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Title and description
st.title("Mitarbeiterabwanderung Vorhersagen App")
st.markdown("""
Willkommen zur Mitarbeiterabwanderung Vorhersagen App! 
Mit dieser App können Sie die Wahrscheinlichkeit der Abwanderung eines Mitarbeiters vorhersagen.
Bitte spezifizieren Sie die Eingabeparameter im Seitenbereich oder laden Sie eine Datei hoch.
""")
st.write('---')

# Load the Dataset
data = pd.read_csv('modified_file.csv')
data.fillna(data.mean(), inplace=True)
target = data.left

string_to_int = {
    'low': 1,
    'medium': 2,
    'high': 3
}

data['gehalt'] = data['gehalt'].map(string_to_int)

features = ["zufriedenheitsgrad", "anzahl_projekte", "durchschnittliche_monatliche_arbeitszeit", "arbeitsunfall", "foerderung_letzte_5_jahre", "gehalt"]

X = data[features]
Y = target

average_montly_hours_min = min(data['durchschnittliche_monatliche_arbeitszeit'])
average_montly_hours_max = max(data['durchschnittliche_monatliche_arbeitszeit'])

# Sidebar input
st.sidebar.header('Eingabeparameter spezifizieren')
st.sidebar.markdown("### Eingabeparameter")

def user_input_features():
    zufriedenheitsgrad = st.sidebar.slider('Zufriedenheitsgrad', 0, 100, help="Der Zufriedenheitsgrad des Mitarbeiters in Prozent.")
    anzahl_projekte = st.sidebar.slider('Anzahl der Projekte', 0, 7, help="Die Anzahl der Projekte, an denen der Mitarbeiter gearbeitet hat.")
    durchschnittliche_monatliche_arbeitszeit = st.sidebar.slider('Durchschnittliche Monatliche Arbeitszeit', average_montly_hours_min, average_montly_hours_max, help="Die durchschnittliche Anzahl der monatlichen Arbeitsstunden.")
    arbeitsunfall = st.sidebar.selectbox('Arbeitsunfall', [0, 1], format_func=lambda x: 'Ja' if x == 1 else 'Nein', help="Ob der Mitarbeiter einen Arbeitsunfall hatte (Ja/Nein).")
    foerderung_letzte_5_jahre = st.sidebar.selectbox('Förderung in den letzten 5 Jahren', [0, 1], format_func=lambda x: 'Ja' if x == 1 else 'Nein', help="Ob der Mitarbeiter in den letzten 5 Jahren befördert wurde (Ja/Nein).")
    gehalt = st.sidebar.selectbox('Gehalt', [1, 2, 3], format_func=lambda x: ['Niedrig', 'Mittel', 'Hoch'][x-1], help="Die Gehaltsstufe des Mitarbeiters (Niedrig, Mittel, Hoch).")
    
    data = {
        'zufriedenheitsgrad': zufriedenheitsgrad,
        'anzahl_projekte': anzahl_projekte,
        'durchschnittliche_monatliche_arbeitszeit': durchschnittliche_monatliche_arbeitszeit,
        'arbeitsunfall': arbeitsunfall,
        'foerderung_letzte_5_jahre': foerderung_letzte_5_jahre,
        'gehalt': gehalt
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel Layout
st.header('Spezifizierte Eingabeparameter')
# Display input parameters in two columns for better layout
col1, col2, col3 = st.columns(3)
col1.metric("Zufriedenheitsgrad", df['zufriedenheitsgrad'][0])
col2.metric("Anzahl der Projekte", df['anzahl_projekte'][0])
col3.metric("Durchschnittliche Monatliche Arbeitszeit", df['durchschnittliche_monatliche_arbeitszeit'][0])

col4, col5, col6 = st.columns(3)
col4.metric("Arbeitsunfall", "Ja" if df['arbeitsunfall'][0] == 1 else "Nein")
col5.metric("Förderung in den letzten 5 Jahren", "Ja" if df['foerderung_letzte_5_jahre'][0] == 1 else "Nein")
col6.metric("Gehalt", ["Niedrig", "Mittel", "Hoch"][df['gehalt'][0] - 1])

st.write('---')

# Build Classifier Model
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)
model_val = model.predict(X_test)
model_accuracy = accuracy_score(y_test, model_val)

# Display model accuracy
st.header('Modellevaluierung')
st.metric(label="Genauigkeit des Modells", value=f"{model_accuracy * 100:.2f}%")
st.write('---')

# Apply Model to Make Prediction
prediction = model.predict(df)

# Display prediction results with color coding
st.header('Vorhersage der Mitarbeiterabwanderung')
prediction_text = 'Der/Die Mitarbeiter/in ist zufrieden und er/sie bleibt bei uns.' if prediction == 0 else 'Der/Die Mitarbeirter/in ist nicht zufrieden und er/sie wird wahrscheinlich gehen.'
prediction_color = 'green' if prediction == 0 else 'red'

st.markdown(f"<h3 style='color:{prediction_color};'>{prediction_text}</h3>", unsafe_allow_html=True)
st.write('---')

# File upload and prediction
st.sidebar.header('Datei-Upload')
uploaded_file = st.sidebar.file_uploader("Laden Sie eine Excel- oder CSV-Datei hoch", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.read_excel(uploaded_file)

        # Check for the required columns
        required_columns = ["zufriedenheitsgrad", "anzahl_projekte", "durchschnittliche_monatliche_arbeitszeit", "arbeitsunfall", "foerderung_letzte_5_jahre", "gehalt"]
        for col in required_columns:
            if col not in input_df.columns:
                st.sidebar.error(f"Die Datei muss die Spalte '{col}' enthalten.")
                st.stop()

        # Map salary column
        input_df['gehalt'] = input_df['gehalt'].map(string_to_int)
        
        predictions = model.predict(input_df[features])
        
        result_mapping = {0: 'Er/Sie bleibt uns', 1: 'Er/Sie wird uns verlassen'}
        input_df['Ergebnis'] = [result_mapping[pred] for pred in predictions]
        
        # Save the result to a CSV file
        result_file = 'predicted_results.csv'
        input_df.to_csv(result_file, index=False)
        
        # Create a download button
        st.sidebar.download_button(
            label="Ergebnisse herunterladen",
            data=input_df.to_csv(index=False).encode('utf-8'),
            file_name=result_file,
            mime='text/csv'
        )
        
        st.header('Hochgeladene Datei und Vorhersagen')
        st.write(input_df)
        st.write('---')

    except Exception as e:
        st.sidebar.error(f"Fehler beim Verarbeiten der Datei: {e}")

# Uncomment the following code if SHAP is installed and configured correctly
# import shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')

# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')
