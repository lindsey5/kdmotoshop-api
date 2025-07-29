from flask import Flask, jsonify
from routes.predict_route import predict_bp
from routes.ai_agent_route import agent_bp
from flask_cors import CORS
from agent.config import model
import os

app = Flask(__name__)

CORS(app,
     origins=["https://kdmotoshop.onrender.com", "http://localhost:5173"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)  # Only if you're sending cookies or auth headers

# Register blueprint
app.register_blueprint(predict_bp)
app.register_blueprint(agent_bp)

@app.route("/run", methods=["POST"])
def run():
    response = model.invoke([{"role": "user", "content": "H"}])
    response.text()

    print(f"Agent response: {response}")

    return jsonify({ "response" : response.content })

# This block only runs locally, not in production
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
