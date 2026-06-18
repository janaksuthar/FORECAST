%%writefile app.py
import streamlit as st
import pandas as pd
from pmdarima import auto_arima # Import auto_arima for model loading and prediction
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for clipping

st.set_page_config(layout='wide')
st.title('Mineral Water Sales and Stock Forecasting App (AutoARIMA)')

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
    model_sales = load_model('model_sales.pkl') # This will now load the AutoARIMA model
    model_soh = load_model('model_soh.pkl')     # This will now load the AutoARIMA model
    st.success('Data and AutoARIMA models loaded successfully!')
except Exception as e:
    st.error(f'Error loading data or models: {e}. Please ensure the Excel file and AutoARIMA model files are in the correct directory.')
    st.stop()

# --- Forecasting Function for AutoARIMA ---
def generate_forecast_arima(model, historical_series, periods):
    # Make predictions for future periods
    forecast_values = model.predict(n_periods=periods)

    # Create a date range for the forecast
    last_date = historical_series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')

    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast_values})
    forecast_df = forecast_df.set_index('ds')
    return forecast_df

# --- Sidebar for User Input ---
st.sidebar.header('Forecasting Settings')
forecast_horizon = st.sidebar.slider('Select Forecast Horizon (Days)', 7, 365, 30)

# --- Main Content ---
st.header('Sales Quantity Forecast (AutoARIMA)')
if st.button('Generate Sales Forecast'):
    with st.spinner('Generating sales forecast...'):
        # Prepare historical data for the ARIMA model
        sales_historical = df_resampled['SALES QTY']

        # Generate forecast
        forecast_sales_arima_df = generate_forecast_arima(model_sales, sales_historical, forecast_horizon)

        st.subheader(f'Sales Quantity Forecast for the next {forecast_horizon} days')

        # Plotting for AutoARIMA (custom plot)
        fig_sales, ax = plt.subplots(figsize=(14, 7))
        ax.plot(sales_historical.index, sales_historical, label='Historical Sales QTY', color='blue')
        ax.plot(forecast_sales_arima_df.index, forecast_sales_arima_df['yhat'], label='AutoARIMA Forecast', color='green', linestyle='--')
        ax.set_title('Sales Quantity Forecast with AutoARIMA')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales Quantity')
        ax.legend()
        st.pyplot(fig_sales)

        st.subheader('Raw Sales Forecast Data')
        st.dataframe(forecast_sales_arima_df.reset_index().rename(columns={'index': 'ds'}))


st.header('Stock on Hand Forecast (AutoARIMA)')
if st.button('Generate Stock on Hand Forecast'):
    with st.spinner('Generating Stock on Hand forecast...'):
        # Prepare historical data for the ARIMA model
        soh_historical = df_resampled['SOH (Stock on Hand)/ Closing Inventory']

        # Generate forecast
        forecast_soh_arima_df = generate_forecast_arima(model_soh, soh_historical, forecast_horizon)

        # Clip negative SOH predictions to zero
        forecast_soh_arima_df['yhat'] = np.maximum(0, forecast_soh_arima_df['yhat'])

        st.subheader(f'Stock on Hand Forecast for the next {forecast_horizon} days')

        # Plotting for AutoARIMA (custom plot)
        fig_soh, ax = plt.subplots(figsize=(14, 7))
        ax.plot(soh_historical.index, soh_historical, label='Historical SOH', color='red')
        ax.plot(forecast_soh_arima_df.index, forecast_soh_arima_df['yhat'], label='AutoARIMA Forecast', color='purple', linestyle='--')
        ax.set_title('Stock on Hand Forecast with AutoARIMA (Negative Values Clipped)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock on Hand')
        ax.legend()
        st.pyplot(fig_soh)

        st.subheader('Raw Stock on Hand Forecast Data')
        st.dataframe(forecast_soh_arima_df.reset_index().rename(columns={'index': 'ds'}))
