# Streamlit app for California House Price Prediction
# Run with: streamlit run streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your dataset
df = pd.read_csv("housing.csv")

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "rf_model.joblib")
print("Model saved!")

st.title('California House Price Predictor (Demo)')

# Load model (the notebook will save rf_model.joblib next to this file)
model_path = 'rf_model.joblib'
try:
    model = joblib.load(model_path)
except Exception as e:
    st.warning('Model not found. Run the notebook script to train and save rf_model.joblib.')
    model = None

data = fetch_california_housing(as_frame=True)
feature_names = data.feature_names

st.sidebar.header('Input features')
user_input = {}
for f in feature_names:
    # provide sensible defaults based on the dataset
    col = float(data.frame[f].median())
    user_input[f] = st.sidebar.number_input(f, value=col, format='%.4f')

if st.sidebar.button('Predict') and model is not None:
    X_user = pd.DataFrame([user_input])
    pred = model.predict(X_user)[0]
    st.write('Predicted median house value (in $100k):', round(pred, 4))
    st.write('Note: Values are the dataset-scaled median house value metric used in sklearn dataset.')

st.write('Dataset sample:')
st.dataframe(data.frame.head())

