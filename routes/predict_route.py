from flask import Blueprint, jsonify
from services.predict_service import forecast_items_qty_sold, predict_future_sales

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['GET', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    forecast_data = predict_future_sales()
    return jsonify(forecast_data)

@predict_bp.route('/predict/items', methods=['GET', 'OPTIONS'])
def predict_items():
    if request.method == 'OPTIONS':
        return '', 200
    forecast_data = forecast_items_qty_sold()
    return jsonify(forecast_data)