import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('production_model.pkl')

# Title of the App
st.title("Meat Production Prediction App")

# User inputs
st.header("Input Parameters")
year = st.number_input("Year", min_value=2023, max_value=2030, step=1)
country = st.selectbox("Country", ["Ireland", "France", "Germany"])
meat_item = st.selectbox("Meat Item", ["Chicken", "Beef", "Goat", "Sheep"])
import_value = st.number_input("Import Value")
export_value = st.number_input("Export Value")

# Prepare the input data
input_data = {
    'Year': [year],
    'Import': [import_value],
    'Export': [export_value],
    f'Country_{country}': [1],
    f'Meat Item_{meat_item}': [1]
}

# Convert to DataFrame and fill missing columns with 0
input_df = pd.DataFrame(input_data)
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0

# Predict production
prediction = model.predict(input_df)[0]

# Display the prediction
st.subheader("Predicted Production")
st.write(f"The predicted production for {country} ({meat_item}) in {year} is {prediction:.2f}")
