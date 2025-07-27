from flask import Blueprint, request, jsonify
from agent.agent import agent_executor, config
from utils.helpers import _build_cors_preflight_response, _corsify_actual_response

agent_bp = Blueprint("agent", __name__)

@agent_bp.route("/api/chat", methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    user_message = request.json.get("message")
    input_message = {"role": "user", "content": user_message}

    result = ""
    for step, metadata in agent_executor.stream(
        {"messages": [input_message]},
        config=config,
        stream_mode="messages"
    ):
        if metadata["langgraph_node"] == "agent" and (text := step.text()):
            result += text

    return _corsify_actual_response(jsonify({"response": result}))