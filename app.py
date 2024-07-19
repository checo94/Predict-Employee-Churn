import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Mitarbeiterabwanderung Vorhersagen",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

st.write("""
# Mitarbeiterabwanderung Vorhersagen App

Diese App prognostiziert die **Mitarbeiterabwanderung**!
""")
st.write('---')

# Load the Dataset
data = pd.read_csv('modified_file.csv')
data.fillna(data.mean(), inplace=True)
target = data.left

string_to_int = {
    'low' : 1,
    'medium' : 2,
    'high' : 3
}

data['salary'] = data['salary'].map(string_to_int)

features = ["satisfaction_level", "number_project", "average_montly_hours", "Work_accident", "promotion_last_5years", "salary"]

X = data[features]
Y = target

average_montly_hours_min = min(data['average_montly_hours'])
average_montly_hours_max = max(data['average_montly_hours'])

# Sidebar
st.sidebar.header('Eingabeparameter spezifizieren')

def user_input_features():
    satisfaction_level = st.sidebar.slider('Zufriedenheitsgrad', 0, 100, help="Der Zufriedenheitsgrad des Mitarbeiters in Prozent.")
    number_project = st.sidebar.slider('Anzahl der Projekte', 0, 7, help="Die Anzahl der Projekte, an denen der Mitarbeiter gearbeitet hat.")
    average_montly_hours = st.sidebar.slider('Durchschnittliche Monatliche Arbeitszeit', average_montly_hours_min, average_montly_hours_max, help="Die durchschnittliche Anzahl der monatlichen Arbeitsstunden.")
    Work_accident = st.sidebar.selectbox('Arbeitsunfall', [0, 1], format_func=lambda x: 'Ja' if x == 1 else 'Nein', help="Ob der Mitarbeiter einen Arbeitsunfall hatte (Ja/Nein).")
    promotion_last_5years = st.sidebar.selectbox('Förderung in den letzten 5 Jahren', [0, 1], format_func=lambda x: 'Ja' if x == 1 else 'Nein', help="Ob der Mitarbeiter in den letzten 5 Jahren befördert wurde (Ja/Nein).")
    salary = st.sidebar.selectbox('Gehalt', [1, 2, 3], format_func=lambda x: ['Niedrig', 'Mittel', 'Hoch'][x-1], help="Die Gehaltsstufe des Mitarbeiters (Niedrig, Mittel, Hoch).")
    
    data = {
        'satisfaction_level': satisfaction_level,
        'number_project': number_project,
        'average_montly_hours': average_montly_hours,
        'Work_accident': Work_accident,
        'promotion_last_5years': promotion_last_5years,
        'salary': salary
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Spezifizierte Eingabeparameter')
st.write(df)
st.write('---')

# Build Classifier Model
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)
model_val = model.predict(X_test)
model_accuracy = accuracy_score(y_test, model_val)

st.header('Modellevaluierung')
st.write(f"Genauigkeit des Modells: **{model_accuracy * 100:.2f}%**")
st.write('---')

# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Vorhersage der Mitarbeiterabwanderung')
if prediction == 1:
    st.markdown("<span style='color:red;'>Der/Die Mitarbeiter/in ist nicht zufrieden und er/sie wird wahrscheinlich gehen.</span>", unsafe_allow_html=True)
else:
    st.markdown("<span style='color:green;'>Der/Die Mitarbeiter/in ist zufrieden und er/sie bleibt bei uns.</span>", unsafe_allow_html=True)
st.write('---')

# Explaining the model's predictions using SHAP values
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
