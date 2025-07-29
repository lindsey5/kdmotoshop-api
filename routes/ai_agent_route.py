from flask import Blueprint, make_response, request, jsonify
from flask_cors import cross_origin
from agent.agent import agent_executor
import uuid

agent_bp = Blueprint("agent", __name__)

@agent_bp.route("/api/chat", methods=['POST', 'OPTIONS'])
@cross_origin(origins=["https://kdmotoshop.onrender.com", "http://localhost:5173"]) 
def chat():
    if request.method == 'OPTIONS':
        # Preflight request
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin')
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response, 200

    # Get thread_id from request or generate a new one
    thread_id = request.json.get("thread_id") or str(uuid.uuid4())
    user_message = request.json.get("message")
    input_message = {"role": "user", "content": user_message}
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = ""
    for step, metadata in agent_executor.stream(
        {"messages": [input_message]},
        config=config,
        stream_mode="messages"
    ):
        if metadata["langgraph_node"] == "agent" and (text := step.text()):
            result += text

    return jsonify({"response": result, "success": True, "thread_id" : thread_id })