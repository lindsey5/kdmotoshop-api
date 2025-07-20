from flask import Flask
from routes.predict_route import predict_bp
from flask_cors import CORS

app = Flask(__name__)

# Register blueprint
app.register_blueprint(predict_bp)

# Allow multiple origins
CORS(app, resources={
    r"/predict/*": {
        "origins": ["http://localhost:5173", "https://kdmotoshop.onrender.com"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

if __name__ == '__main__':
    app.run(debug=True)

    @app.route('/')
    def home():
        return 'Hello from KDMotoshop on Render!'