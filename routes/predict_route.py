from flask import Blueprint, jsonify
from services.predict_service import predict_future_sales

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['GET'])
def predict():
    forecast_data = predict_future_sales()
    return jsonify(forecast_data)