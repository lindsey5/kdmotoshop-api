import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load('model/kdmotoshop_xgb_model.pkl')

# Forecasting function
def forecast_future(model, daily_sales, n_days=30):
    last_date = daily_sales['DATE'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days, freq='D')
    
    recent_sales = daily_sales['SOLD PRICE'].values.copy()
    recent_ema = daily_sales['ema_7'].iloc[-1]
    alpha = 2 / (7 + 1)  # Smoothing factor for EMA

    future_predictions = []

    for i, future_date in enumerate(future_dates):
        dayofweek = future_date.dayofweek
        day = future_date.day
        month = future_date.month
        is_weekend = 1 if dayofweek >= 5 else 0
        
        lag_1 = recent_sales[-1] if len(recent_sales) >= 1 else 0
        lag_2 = recent_sales[-2] if len(recent_sales) >= 2 else 0
        lag_3 = recent_sales[-3] if len(recent_sales) >= 3 else 0
        lag_7 = recent_sales[-7] if len(recent_sales) >= 7 else recent_sales[-1]
        rolling_mean_7 = np.mean(recent_sales[-7:]) if len(recent_sales) >= 7 else np.mean(recent_sales)
        rolling_std_7 = np.std(recent_sales[-7:]) if len(recent_sales) >= 7 else 0
        diff_7 = lag_1 - lag_7 if len(recent_sales) >= 7 else 0

        # Update EMA manually
        ema_7 = (lag_1 * alpha) + (recent_ema * (1 - alpha))

        features = np.array([[dayofweek, day, month, is_weekend,
                              lag_1, lag_2, lag_3, lag_7,
                              rolling_mean_7, rolling_std_7,
                              diff_7, ema_7]])

        prediction = model.predict(features)[0]
        future_predictions.append(prediction)
        
        recent_sales = np.append(recent_sales, prediction)
        recent_ema = ema_7  # update EMA for next day

    return future_dates, future_predictions

# Main function
def predict_future_sales():
    try:
        # Load and process the sales data
        file_path = 'data/KD INVENTORY & SALES.xlsx'
        sheet_name = 'E-COM January-June 2025 Sales'
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df = df.dropna(subset=['DATE', 'SOLD PRICE'])
        daily_sales = df.groupby(df['DATE'].dt.date)['SOLD PRICE'].sum().reset_index()
        daily_sales.columns = ['DATE', 'SOLD PRICE']
        daily_sales['DATE'] = pd.to_datetime(daily_sales['DATE'])
        daily_sales = daily_sales.sort_values('DATE').reset_index(drop=True)

        # Compute 7-day EMA
        daily_sales['ema_7'] = daily_sales['SOLD PRICE'].ewm(span=7, adjust=False).mean()

        # Forecast future sales
        future_dates, future_preds = forecast_future(model, daily_sales, n_days=30)

        return {
            'forecast':[float(pred) for pred in future_preds],
            'dates': future_dates.strftime('%Y-%m-%d').tolist(),
            'success': True,
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
        }