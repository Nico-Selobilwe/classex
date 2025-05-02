import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Load all models
@st.cache_resource
def load_models():
    models = {
        'Tesla (TSLA)': joblib.load('tesla_sarimax_model.joblib'),
        'Apple (AAPL)': joblib.load('aapl_sarimax_model.joblib'),
        'Boeing (BA)': joblib.load('ba_sarimax_model.joblib'),
        'Amazon (AMZN)': joblib.load('amzn_sarimax_model.joblib'),
        'Disney (DIS)': joblib.load('dis_sarimax_model.joblib')
    }
    return models

models = load_models()

# Dashboard layout
st.title('Stock Price Prediction Dashboard')
st.markdown("""
Predict stock prices for 1, 3, and 7 days using SARIMAX models trained on historical data.
""")

# Sidebar controls
st.sidebar.header('Settings')
selected_company = st.sidebar.selectbox('Select Company', list(models.keys()))
forecast_horizon = st.sidebar.radio('Forecast Horizon', [1, 3, 7], horizontal=True)

# Main prediction area
st.header(f"{selected_company} Price Prediction")

# Get the selected model
model_data = models[selected_company]

# Input for sentiment values
st.subheader("Enter Sentiment Scores")
st.markdown("""
Provide sentiment scores (-1 to 1) for each day, where:
- **-1** = Very Negative  
- **0** = Neutral  
- **1** = Very Positive
""")

# Create dynamic input fields based on horizon
cols = st.columns(forecast_horizon)
sentiment_values = []

for i in range(forecast_horizon):
    with cols[i]:
        day_label = (datetime.now() + timedelta(days=i)).strftime('%b %d')
        val = st.number_input(
            f"Day {i+1} ({day_label})",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            key=f"sentiment_{i}"
        )
        sentiment_values.append(val)

# Prediction button
if st.button('Predict Prices', type='primary'):
    # Prepare exogenous data
    exog_data = pd.DataFrame({'compound': sentiment_values})
    
    try:
        # Get forecast
        forecast = model_data['model'].get_forecast(
            steps=forecast_horizon,
            exog=exog_data
        )
        
        # Get predictions and confidence intervals
        preds = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Create results table
        results = pd.DataFrame({
            'Date': pd.date_range(start=datetime.now(), periods=forecast_horizon).strftime('%Y-%m-%d'),
            'Predicted Price': [f"${x:.2f}" for x in preds],
            'Low Estimate': [f"${x:.2f}" for x in conf_int.iloc[:, 0]],
            'High Estimate': [f"${x:.2f}" for x in conf_int.iloc[:, 1]],
            'Sentiment': sentiment_values
        })
        
        # Display results
        st.subheader("Prediction Results")
        st.dataframe(results.style.format({'Sentiment': '{:.1f}'}))
        
        # Show chart
        st.subheader("Price Forecast")
        chart_data = pd.DataFrame({
            'Prediction': preds,
            'Lower Bound': conf_int.iloc[:, 0],
            'Upper Bound': conf_int.iloc[:, 1]
        }, index=results['Date'])
        
        st.line_chart(chart_data)
        
        # Model info
        with st.expander("Model Information"):
            st.write(f"**Last Training Date:** {model_data['last_training_date']}")
            st.write(f"**Model Order (p,d,q):** {model_data['order']}")
            st.write(f"**Seasonal Order (P,D,Q,m):** {model_data['seasonal_order']}")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Add footer
st.markdown("---")
st.caption("Note: Predictions are based on historical patterns and may not account for unexpected market events.")