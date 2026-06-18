import streamlit as st
import pandas as pd
from prophet import Prophet
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

file_path = '/content/Mineral Water- 2 years sales and stock data.xlsx'

try:
    df_resampled = load_data(file_path)
    model_sales = load_model('model_sales.pkl')
    model_soh = load_model('model_soh.pkl')
    st.success('Data and models loaded successfully!')
except Exception as e:
    st.error(f'Error loading data or models: {e}. Please ensure the Excel file and model files are in the correct directory.')
    st.stop()

# --- Forecasting Function ---
def generate_forecast(model, periods, include_history=True):
    future = model.make_future_dataframe(periods=periods, include_history=include_history)
    forecast = model.predict(future)
    return forecast

# --- Sidebar for User Input ---
st.sidebar.header('Forecasting Settings')
forecast_horizon = st.sidebar.slider('Select Forecast Horizon (Days)', 7, 365, 30)

# --- Main Content ---

st.header('Sales Quantity Forecast')
if st.button('Generate Sales Forecast'):
    with st.spinner('Generating sales forecast...'):
        forecast_sales = generate_forecast(model_sales, periods=forecast_horizon)
        st.subheader(f'Sales Quantity Forecast for the next {forecast_horizon} days')
        fig_sales = model_sales.plot(forecast_sales)
        st.pyplot(fig_sales)

        st.subheader('Sales Forecast Components')
        fig_components_sales = model_sales.plot_components(forecast_sales)
        st.pyplot(fig_components_sales)

        st.subheader('Raw Sales Forecast Data')
        st.dataframe(forecast_sales[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_horizon))

st.header('Stock on Hand Forecast')
if st.button('Generate Stock on Hand Forecast'):
    with st.spinner('Generating Stock on Hand forecast...'):
        forecast_soh = generate_forecast(model_soh, periods=forecast_horizon)
        st.subheader(f'Stock on Hand Forecast for the next {forecast_horizon} days')
        fig_soh = model_soh.plot(forecast_soh)
        st.pyplot(fig_soh)

        st.subheader('Stock on Hand Forecast Components')
        fig_components_soh = model_soh.plot_components(forecast_soh)
        st.pyplot(fig_components_soh)

        st.subheader('Raw Stock on Hand Forecast Data')
        st.dataframe(forecast_soh[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_horizon))
