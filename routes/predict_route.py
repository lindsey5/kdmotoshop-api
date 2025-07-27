from flask import Blueprint, jsonify, request
from services.predict_service import forecast_items_qty_sold, predict_future_sales
from utils.helpers import _build_cors_preflight_response, _corsify_actual_response

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/api/predict', methods=['GET', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    forecast_data = predict_future_sales()
    return _corsify_actual_response(jsonify(forecast_data))

@predict_bp.route('/api/predict/items', methods=['GET', 'OPTIONS'])
def predict_items():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    forecast_data = forecast_items_qty_sold()
    return _corsify_actual_response(jsonify(forecast_data))
    