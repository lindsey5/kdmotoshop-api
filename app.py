from flask import Flask
from routes.predict_route import predict_bp
from flask_cors import CORS

app = Flask(__name__)

# Register blueprint
app.register_blueprint(predict_bp)

CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

if __name__ == '__main__':
    app.run(debug=True)