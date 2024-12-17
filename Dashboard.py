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
country = st.selectbox("Country", ["France", "Germany", "Ireland"])
meat_item = st.selectbox("Meat Item", ["Meat_Cattle", "Meat_Chicken", "Meat_Goat", "Meat_Pig", "Meat_Sheep"])
import_value = st.number_input("Import Value")
export_value = st.number_input("Export Value")

# Prepare the input data
input_data = {
    'Year': [year],
    'Export': [export_value],
    'Import': [import_value],
    'Country_France': [0],
    'Country_Germany': [0],
    'Country_Ireland': [0],
    'Item_Meat_Cattle': [0],
    'Item_Meat_Chicken': [0],
    'Item_Meat_Goat': [0],
    'Item_Meat_Pig': [0],
    'Item_Meat_Sheep': [0]
}

# Set the selected country and meat item
input_data[f'Country_{country}'] = [1]
input_data[f'Item_{meat_item}'] = [1]

# Convert to DataFrame
input_df = pd.DataFrame(input_data)

# Ensure columns are in the correct order as expected by the model
input_df = input_df[model.feature_names_in_]

# Predict production
prediction = model.predict(input_df)[0]

# Display the prediction
st.subheader("Predicted Production")
st.write(f"The predicted production for {country} ({meat_item}) in {year} is {prediction:.2f}")
