import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set page config
st.set_page_config(page_title="Employee Churn Prediction", page_icon="👥", layout="wide")

# Custom CSS to improve the design
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
.medium-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-font">Employee Churn Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="medium-font">This app predicts the likelihood of employee churn based on various factors.</p>', unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('modified_file.csv')
    data.fillna(data.mean(), inplace=True)
    string_to_int = {'low': 1, 'medium': 2, 'high': 3}
    data['salary'] = data['salary'].map(string_to_int)
    return data

data = load_data()
features = ["satisfaction_level", "number_project", "average_montly_hours", "Work_accident", "promotion_last_5years", "salary"]
X = data[features]
Y = data.left

# Sidebar
st.sidebar.header('Input Parameters')

def user_input_features():
    satisfaction_level = st.sidebar.slider('Satisfaction Level', 0.0, 1.0, 0.5)
    number_project = st.sidebar.slider('Number of Projects', 2, 7, 4)
    average_montly_hours = st.sidebar.slider('Average Monthly Hours', int(X['average_montly_hours'].min()), int(X['average_montly_hours'].max()), int(X['average_montly_hours'].mean()))
    work_accident = st.sidebar.selectbox('Work Accident', [0, 1])
    promotion_last_5years = st.sidebar.selectbox('Promotion in Last 5 Years', [0, 1])
    salary = st.sidebar.selectbox('Salary', ['Low', 'Medium', 'High'])
    
    salary_dict = {'Low': 1, 'Medium': 2, 'High': 3}
    
    data = {
        'satisfaction_level': satisfaction_level,
        'number_project': number_project,
        'average_montly_hours': average_montly_hours,
        'Work_accident': work_accident,
        'promotion_last_5years': promotion_last_5years,
        'salary': salary_dict[salary]
    }
    return pd.DataFrame(data, index=[0])

# Main panel
col1, col2 = st.columns(2)

with col1:
    st.subheader('Specified Input Parameters')
    df = user_input_features()
    st.write(df)

# Model building
@st.cache_resource
def build_model():
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = build_model()

# Make prediction
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

with col2:
    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.error('The employee is likely to leave.')
    else:
        st.success('The employee is likely to stay.')
    
    st.write(f'Probability of staying: {prediction_proba[0][0]:.2f}')
    st.write(f'Probability of leaving: {prediction_proba[0][1]:.2f}')

# Model performance
st.subheader('Model Performance')
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(plt)

# Feature Importance
st.subheader('Feature Importance')
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
st.pyplot(plt)

# Classification Report
st.subheader('Classification Report')
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.table(df_report)
