from flask import Blueprint, jsonify
from flask_cors import cross_origin
from services.predict_service import forecast_items_qty_sold, predict_future_sales

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/api/predict', methods=['GET', 'OPTIONS'])
def predict():
    forecast_data = predict_future_sales()
    return jsonify(forecast_data)

@predict_bp.route('/api/predict/items', methods=['GET', 'OPTIONS'])
@cross_origin()
def predict_items():
    forecast_data = forecast_items_qty_sold()
    return jsonify(forecast_data)
    