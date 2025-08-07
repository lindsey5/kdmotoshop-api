import calendar
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

def forecast_next_days(model, historical_df, year, month):
    """
    Forecast next `num_days` using trained model and past features.
    """
    df_last_date = pd.to_datetime(historical_df['DATE'].iloc[-1])

    # Last day of the target month
    _, last_day = calendar.monthrange(year, month)
    forecast_end_date = datetime(year, month, last_day)

    # Start forecasting from the next day after last known date
    forecast_start_date = df_last_date + pd.Timedelta(days=1)

    # Compute number of days to forecast
    num_days = (forecast_end_date - forecast_start_date).days + 1  # +1 to include end date

    # Generate forecast dates
    forecast_dates = pd.date_range(start=forecast_start_date, periods=num_days)
    forecast_df = []

    temp_df = historical_df.copy()

    for date in forecast_dates:
        dayofweek = date.dayofweek
        day = date.day
        month = date.month
        is_weekend = 1 if dayofweek >= 5 else 0

        lag_1 = temp_df['SOLD PRICE'].iloc[-1]
        lag_2 = temp_df['SOLD PRICE'].iloc[-2]
        lag_3 = temp_df['SOLD PRICE'].iloc[-3]
        lag_7 = temp_df['SOLD PRICE'].iloc[-7] if len(temp_df) >= 7 else temp_df['SOLD PRICE'].mean()
        lag_30 = temp_df['SOLD PRICE'].iloc[-30] if len(temp_df) >= 30 else temp_df['SOLD PRICE'].mean()


        rolling = temp_df['SOLD PRICE'].rolling(window=7, min_periods=1)
        rolling_mean_7 = rolling.mean().iloc[-1]

        rolling = temp_df['SOLD PRICE'].rolling(window=30, min_periods=1)
        rolling_mean_30 = rolling.mean().iloc[-1]

        diff_30 = temp_df['SOLD PRICE'].iloc[-1] - temp_df['SOLD PRICE'].iloc[-31] if len(temp_df) >= 31 else 0

        features = np.array([
            dayofweek, day, month, is_weekend,
            lag_1, lag_2, lag_3, lag_7,
            lag_30,
            rolling_mean_7,
            rolling_mean_30,
            diff_30,
        ]).reshape(1, -1)

        predicted_price = model.predict(features)[0]
        predicted_price = max(predicted_price, 0)  # Avoid negative predictions

        # Append prediction
        forecast_df.append({'DATE': date, 'PREDICTED_SALES': predicted_price})

        # Append to temp_df for next lag feature generation
        temp_df = pd.concat([
            temp_df,
            pd.DataFrame([{'DATE': date, 'SOLD PRICE': predicted_price}])
        ], ignore_index=True)

    forecast_df = pd.DataFrame(forecast_df)
    forecast_df = forecast_df[
        (forecast_df['DATE'].dt.month == month) & 
        (forecast_df['DATE'].dt.year == year)
    ]
    return forecast_df

def predict_future_sales(month, year):
    try:
        df = loadDataset()

        daily_sales = df.groupby('DATE')['SOLD PRICE'].sum().reset_index()
        daily_sales.set_index('DATE', inplace=True)
        daily_sales.sort_index(inplace=True)

        # Forecast future sales
        forecast_results = forecast_next_days(model, daily_sales.reset_index(), year, month)

        return {
            'forecast':[float(pred) for pred in forecast_results['PREDICTED_SALES']],
            'forecast_dates': forecast_results['DATE'].dt.strftime('%Y-%m-%d').tolist(),
            'actual_sales' : [float(sales) for sales in daily_sales['SOLD PRICE']],
            'dates' :  daily_sales.reset_index()['DATE'].dt.strftime('%Y-%m-%d').tolist(),
            'success': True,
            'month' : 'July 2025'
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

def forecast_items_qty_sold():
    df = loadDataset()
    df['MONTH'] = df['DATE'].dt.month
    df['YEAR'] = df['DATE'].dt.year

    monthly_data = df.groupby(['ITEM DESCRIPTION', 'YEAR', 'MONTH']).agg({
        'QTY': 'sum',
        'SOLD PRICE': 'mean',
    }).reset_index()

    monthly_data['QUARTER'] = monthly_data['MONTH'].apply(lambda x: ((x - 1) // 3) + 1)
    data = create_global_features(monthly_data)

    today = datetime.today()

    current_month = today.month
    current_year = today.year
    current_quarter = ((current_month - 1) // 3) + 1

    feature_columns = [
        'ITEM_ENCODED','TIME_IDX','MONTH','QUARTER',
        'QTY_LAG1','QTY_LAG2','QTY_LAG3',
        'QTY_ROLLING_3','QTY_ROLLING_6','QTY_RELATIVE',
        'MONTH_SIN','MONTH_COS','QUARTER_SIN','QUARTER_COS',
    ]

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
            'price': round(item_df['SOLD PRICE'].iloc[-1], 2),
            'predicted_qty': round(pred, 0),
            'sales': round(item_df['SOLD PRICE'].iloc[-1] * round(pred, 0), 2),
            'months_of_data': int(len(item_df)),
            'last_month_qty': int(item_df['QTY'].iloc[-1]),
            'predicted_month': f"{current_month:02d}",
        })
    return {
        'forecast': predictions,
        'success': True,
        'month': 'July 2025'
    }
