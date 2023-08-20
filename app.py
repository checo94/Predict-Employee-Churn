import streamlit as st
import pandas as pd
# import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Mitarbeiterabwanderung vorhersagen App

Diese App prognostiziert die **Mitarbeiterabwanderung**!
""")
st.write('---')

# Loads The Dataset
data = pd.read_csv('HCM_Employee_Churn.csv')
data.fillna(data.mean(), inplace=True)
target = data.left

features = ["satisfaction_level", "number_project", "average_montly_hours", "Work_accident", "promotion_last_5years", "salary"]

X = data[features]
Y = target

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    satisfaction_level = st.sidebar.slider('Zufriedenheitsgrad', 0, 100)
    number_project = st.sidebar.slider('Anzahl der Projekte', 0, 7)
    average_montly_hours = st.sidebar.slider('Durchschnittliche Monatliche Arbeitszeit', 0, 300)
    Work_accident = st.sidebar.slider('Arbeitsunfall', 0, 1)
    promotion_last_5years = st.sidebar.slider('Förderung', 0, 1)
    salary = st.sidebar.slider('Gehalt', 1, 3)
    data = {'satisfaction_level': satisfaction_level,
            'number_project': number_project,
            'average_montly_hours': average_montly_hours,
            'Work_accident': Work_accident,
            'promotion_last_5years': promotion_last_5years,
            'salary': salary}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Spezifizierte Eingabeparameter')
st.write(df)
st.write('---')

# Build Classifier Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, Y)
# Apply Model to Make Prediction
model_val = model.predict(X_test)
model_accuracy = accuracy_score(y_test, model_val)
print(model_accuracy)
prediction = model.predict(df)

st.header('Mitarbeiterabwanderung Ergebniss')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
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
