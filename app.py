import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for clipping
import os

st.set_page_config(layout='wide')
st.title('Mineral Water Sales and Stock Forecasting App')

# --- Load Data ---
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path)
    df['BUSINESS_DATE'] = pd.to_datetime(df['BUSINESS_DATE'])
    df = df.set_index('BUSINESS_DATE').sort_index()
    df_resampled = df[['SALES QTY', 'SOH (Stock on Hand)/ Closing Inventory']].resample('D').sum()
    return df_resampled

# --- Load Models ---
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

file_path = 'Mineral Water- 2 years sales and stock data.xlsx'

try:
    # Check if files exist
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file '{file_path}' not found in the current directory.")
    if not os.path.exists('model_sales.pkl'):
        raise FileNotFoundError("'model_sales.pkl' not found in the current directory.")
    if not os.path.exists('model_soh.pkl'):
        raise FileNotFoundError("'model_soh.pkl' not found in the current directory.")
    
    df_resampled = load_data(file_path)
    model_sales = load_model('model_sales.pkl')
    model_soh = load_model('model_soh.pkl')
    st.success('Data and models loaded successfully!')
except FileNotFoundError as e:
    st.error(f'File not found: {e}')
    st.info('Please ensure the following files are in the app directory:\n- Mineral Water- 2 years sales and stock data.xlsx\n- model_sales.pkl\n- model_soh.pkl')
    st.stop()
except Exception as e:
    st.error(f'Error loading data or models: {e}')
    st.stop()

# --- Forecasting Function for Prophet ---
def generate_forecast_prophet(model, periods):
    # Make future dataframe for Prophet
    future = model.make_future_dataframe(periods=periods)
    # Generate forecast
    forecast = model.predict(future)
    return forecast

# --- Sidebar for User Input ---
st.sidebar.header('Forecasting Settings')
forecast_horizon = st.sidebar.slider('Select Forecast Horizon (Days)', 7, 365, 30)

# --- Main Content ---
st.header('Sales Quantity Forecast')
if st.button('Generate Sales Forecast'):
    with st.spinner('Generating sales forecast...'):
        # Generate forecast
        forecast_sales = generate_forecast_prophet(model_sales, forecast_horizon)
        
        # Clip negative sales predictions to zero
        forecast_sales['yhat'] = np.maximum(0, forecast_sales['yhat'])

        st.subheader(f'Sales Quantity Forecast for the next {forecast_horizon} days')

        # Plotting for Prophet
        fig_sales = model_sales.plot(forecast_sales)
        plt.grid(True, alpha=0.3)
        st.pyplot(fig_sales)

        st.subheader('Sales Forecast Components')
        fig_components_sales = model_sales.plot_components(forecast_sales)
        st.pyplot(fig_components_sales)

        st.subheader('Raw Sales Forecast Data')
        st.dataframe(forecast_sales[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_horizon))


st.header('Stock on Hand Forecast')
if st.button('Generate Stock on Hand Forecast'):
    with st.spinner('Generating Stock on Hand forecast...'):
        # Generate forecast
        forecast_soh = generate_forecast_prophet(model_soh, forecast_horizon)

        # Clip negative SOH predictions to zero
        forecast_soh['yhat'] = np.maximum(0, forecast_soh['yhat'])

        st.subheader(f'Stock on Hand Forecast for the next {forecast_horizon} days')

        # Plotting for Prophet
        fig_soh = model_soh.plot(forecast_soh)
        plt.grid(True, alpha=0.3)
        st.pyplot(fig_soh)

        st.subheader('Stock on Hand Forecast Components')
        fig_components_soh = model_soh.plot_components(forecast_soh)
        st.pyplot(fig_components_soh)

        st.subheader('Raw Stock on Hand Forecast Data')
        st.dataframe(forecast_soh[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_horizon))
