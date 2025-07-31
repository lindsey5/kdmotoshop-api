from flask import Blueprint, jsonify
from flask_cors import cross_origin
from services.predict_service import forecast_items_qty_sold, predict_future_sales

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/api/predict', methods=['GET', 'OPTIONS'])
def predict():
    try:
        forecast_data = predict_future_sales()
        return jsonify(forecast_data)
    except Exception as e:
            print("Error", str(e))
            return jsonify({"error": "Internal Server Error"}), 500

@predict_bp.route('/api/predict/items', methods=['GET', 'OPTIONS'])
def predict_items():
    try:
        forecast_data = forecast_items_qty_sold()
        return jsonify(forecast_data)
    except Exception as e:
                print("Error", str(e))
                return jsonify({"error": "Internal Server Error"}), 500
    