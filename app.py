from flask import Flask, request
from routes.predict_route import predict_bp
from routes.ai_agent_route import agent_bp
from flask_cors import CORS
import os

app = Flask(__name__)

CORS(app, 
     origins=["http://localhost:5173", "https://kdmotoshop.onrender.com"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

@app.before_request
def handle_preflight():
    if request.method.lower() == 'options':
        response = app.make_default_options_response()
        headers = response.headers
        headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
        headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept'
        return response

# Register blueprint
app.register_blueprint(predict_bp)
app.register_blueprint(agent_bp)

# This route must be outside the __main__ block
@app.route('/')
def home():
    return 'Hello from KDMotoshop on Render!'

# This block only runs locally, not in production
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
