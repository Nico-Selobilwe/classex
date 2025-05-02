import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Load all models
@st.cache_resource
def load_models():
    models = {
        'TSLA': joblib.load('tesla_sarimax_model.joblib'),
        'AAPL': joblib.load('aapl_sarimax_model.joblib'),
        'BA': joblib.load('ba_sarimax_model.joblib'),
        'AMZN': joblib.load('amzn_sarimax_model.joblib'),
        'DIS': joblib.load('dis_sarimax_model.joblib')
    }
    return models

models = load_models()

# Dashboard layout
st.title('Stock Price Prediction Dashboard')
st.markdown("""
Predict stock prices using SARIMAX models trained on historical price and sentiment data.
""")

# Sidebar controls
st.sidebar.header('Prediction Settings')
selected_stock = st.sidebar.selectbox('Select Stock', list(models.keys()))
forecast_days = st.sidebar.slider('Forecast Horizon (days)', 1, 14, 3)

# Suggested optimal inputs based on historical patterns
suggested_inputs = {
    'TSLA': {'compound': [0.5, 0.4, 0.3], 'note': "Tesla responds well to moderately positive sentiment"},
    'AAPL': {'compound': [0.3, 0.2, 0.1], 'note': "Apple shows stable performance with neutral sentiment"},
    'BA': {'compound': [0.6, 0.5, 0.4], 'note': "Boeing benefits from strong positive sentiment"},
    'AMZN': {'compound': [0.4, 0.3, 0.2], 'note': "Amazon performs well with slightly positive sentiment"},
    'DIS': {'compound': [0.7, 0.6, 0.5], 'note': "Disney is highly sensitive to positive sentiment"}
}

# Input section
st.subheader(f"Input Parameters for {selected_stock}")
st.info(f"Suggested: {suggested_inputs[selected_stock]['note']}")

default_values = suggested_inputs[selected_stock]['compound']
compound_input = []

cols = st.columns(forecast_days)
for i in range(forecast_days):
    with cols[i]:
        day = (datetime.now() + timedelta(days=i)).strftime('%b %d')
        val = st.number_input(
            f"Day {i+1} ({day}) Sentiment",
            min_value=-1.0,
            max_value=1.0,
            value=default_values[i] if i < len(default_values) else 0.0,
            step=0.1,
            help="Sentiment score (-1 to 1) where 1=most positive"
        )
        compound_input.append(val)

# Make prediction
if st.button('Predict Stock Prices'):
    model_data = models[selected_stock]
    exog_data = pd.DataFrame({'compound': compound_input[:forecast_days]})
    
    try:
        forecast = model_data['model'].get_forecast(
            steps=forecast_days,
            exog=exog_data
        )
        
        # Display results
        st.subheader(f"{selected_stock} {forecast_days}-Day Price Prediction")
        
        preds = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Create results table
        results = pd.DataFrame({
            'Date': pd.date_range(start=datetime.now(), periods=forecast_days).strftime('%Y-%m-%d'),
            'Predicted Price': [f"${x:.2f}" for x in preds],
            'Low Estimate': [f"${x:.2f}" for x in conf_int.iloc[:, 0]],
            'High Estimate': [f"${x:.2f}" for x in conf_int.iloc[:, 1]],
            'Sentiment Input': compound_input[:forecast_days]
        })
        
        st.dataframe(results.style.format(None, subset=['Sentiment Input']))
        
        # Show chart
        chart_data = pd.DataFrame({
            'Prediction': preds,
            'Lower Bound': conf_int.iloc[:, 0],
            'Upper Bound': conf_int.iloc[:, 1]
        }, index=results['Date'])
        
        st.line_chart(chart_data)
        
        # Model info
        with st.expander("Model Details"):
            st.write(f"**Last Trained:** {model_data['last_training_date']}")
            st.write(f"**Order (p,d,q):** {model_data['order']}")
            st.write(f"**Seasonal Order (P,D,Q,m):** {model_data['seasonal_order']}")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Best practices section
st.sidebar.markdown("""
### Best Input Suggestions:
1. **Use recent sentiment trends** - Check news sentiment for the stock
2. **For growth stocks (TSLA, AMZN):** Moderate positive (0.3-0.6)
3. **For stable stocks (AAPL):** Neutral to slightly positive (0.1-0.3)
4. **For cyclical stocks (BA, DIS):** Strong positive (0.5-0.8)
5. **Avoid extreme values** (-1 or 1) unless major news event
""")