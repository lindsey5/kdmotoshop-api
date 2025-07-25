from flask import jsonify
import pandas as pd
import joblib
import numpy as np
from sklearn.calibration import LabelEncoder
from datetime import datetime

# Load model
model = joblib.load('model/kdmotoshop_xgb_model.pkl')
model_2 = joblib.load('model/kd_xgb_qty_sold.pkl')

def loadDataset():
    # Load and process the sales data
    file_path = 'data/KD INVENTORY & SALES.xlsx'
    sheet_name = 'E-COM January-June 2025 Sales'
    return pd.read_excel(file_path, sheet_name=sheet_name)

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


def predict_future_sales():
    try:
        df = loadDataset()

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
    

def create_global_features(data):
    item_encoder = LabelEncoder()
    data['ITEM_ENCODED'] = item_encoder.fit_transform(data['ITEM DESCRIPTION'])
    data = data.sort_values(['ITEM DESCRIPTION', 'YEAR', 'MONTH'])
    data['TIME_IDX'] = (data['YEAR'] - data['YEAR'].min()) * 12 + data['MONTH']

    item_stats = data.groupby('ITEM DESCRIPTION').agg({
        'QTY': ['mean', 'std', 'max', 'min'],
    })
    item_stats.columns = [f'ITEM_{c[0]}_{c[1]}' for c in item_stats.columns]
    item_stats = item_stats.reset_index()
    data = data.merge(item_stats, on='ITEM DESCRIPTION', how='left')

    print(item_stats.columns)

    item_mean_map = data.set_index('ITEM DESCRIPTION')['ITEM_QTY_mean'].to_dict()

    lagged_list = []
    for item, item_df in data.groupby('ITEM DESCRIPTION'):
        item_df = item_df.copy().sort_values(['YEAR', 'MONTH'])
        item_df['QTY_LAG1'] = item_df['QTY'].shift(1)
        item_df['QTY_LAG2'] = item_df['QTY'].shift(2)
        item_df['QTY_LAG3'] = item_df['QTY'].shift(3)
        item_df['QTY_ROLLING_3'] = item_df['QTY'].rolling(3, min_periods=1).mean()
        item_df['QTY_ROLLING_6'] = item_df['QTY'].rolling(6, min_periods=1).mean()
        item_df['QTY_RELATIVE'] = item_df['QTY'] / item_mean_map[item]
        lagged_list.append(item_df)

    data_with_lags = pd.concat(lagged_list, ignore_index=True)
    data_with_lags['MONTH_SIN'] = np.sin(2 * np.pi * data_with_lags['MONTH'] / 12)
    data_with_lags['MONTH_COS'] = np.cos(2 * np.pi * data_with_lags['MONTH'] / 12)
    data_with_lags['QUARTER_SIN'] = np.sin(2 * np.pi * data_with_lags['QUARTER'] / 4)
    data_with_lags['QUARTER_COS'] = np.cos(2 * np.pi * data_with_lags['QUARTER'] / 4)

    monthly_totals = data_with_lags.groupby(['YEAR', 'MONTH']).agg({
        'QTY': 'sum'
    }).rename(columns={'QTY': 'MARKET_QTY'}).reset_index()

    data_with_lags = data_with_lags.merge(monthly_totals, on=['YEAR', 'MONTH'], how='left')
    data_with_lags = data_with_lags.fillna(0)

    return data_with_lags

feature_columns = [
    'ITEM_ENCODED','TIME_IDX','MONTH','QUARTER',
    'QTY_LAG1','QTY_LAG2','QTY_LAG3',
    'QTY_ROLLING_3','QTY_ROLLING_6','QTY_RELATIVE',
    'MONTH_SIN','MONTH_COS','QUARTER_SIN','QUARTER_COS',
]

def forecast_items_qty_sold():
    df = loadDataset()
    df['MONTH'] = df['DATE'].dt.month
    df['YEAR'] = df['DATE'].dt.year

    monthly_data = df.groupby(['ITEM DESCRIPTION', 'YEAR', 'MONTH']).agg({
        'QTY': 'sum',
    }).reset_index()

    monthly_data['QUARTER'] = monthly_data['MONTH'].apply(lambda x: ((x - 1) // 3) + 1)
    data = create_global_features(monthly_data)

    today = datetime.today()

    current_month = today.month
    current_year = today.year
    current_quarter = ((current_month - 1) // 3) + 1

    predictions = []

    # Group by item
    for item, item_df in data.groupby('ITEM DESCRIPTION'):
        item_df = item_df.sort_values(['YEAR', 'MONTH'])

        # Skip if less than 2 months of data
        if len(item_df) < 2:
            continue

        # Get features from last row
        last = item_df.iloc[-1:].copy()
        feat = last[feature_columns].iloc[0:1].copy()

        feat['MONTH'] = (current_year - 2025) * 12 + (current_month - 1)
        feat['MONTH_SIN'] = np.sin(2 * np.pi * current_month / 12)
        feat['MONTH_COS'] = np.cos(2 * np.pi * current_month / 12)
        feat['QUARTER'] = current_quarter
        feat['QUARTER_SIN'] = np.sin(2 * np.pi * current_quarter / 4)
        feat['QUARTER_COS'] = np.cos(2 * np.pi * current_quarter / 4)

        # Lag and rolling features
        feat['QTY_LAG1'] = item_df['QTY'].iloc[-1]
        feat['QTY_LAG2'] = item_df['QTY'].iloc[-2]
        feat['QTY_LAG3'] = item_df['QTY'].iloc[-3] if len(item_df) >= 3 else feat['QTY_LAG2']
        feat['QTY_ROLLING_3'] = item_df['QTY'].tail(3).mean()
        feat['QTY_ROLLING_6'] = item_df['QTY'].tail(6).mean()

        # Make prediction
        pred =float(max(0, model_2.predict(feat).item()))

        predictions.append({
            'item': str(item),
            'predicted_qty': round(pred, 2),
            'months_of_data': int(len(item_df)),
            'last_month_qty': int(item_df['QTY'].iloc[-1]),
            'predicted_month': f"{current_month:02d}",
        })
    return {
        'forecast': predictions,
        'success': True,
    }
