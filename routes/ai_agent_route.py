from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from agent.agent import agent_executor, config

agent_bp = Blueprint("agent", __name__)

@agent_bp.route("/api/chat", methods=['POST', 'OPTIONS'])
@cross_origin()
def chat():
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

    return jsonify({"response": result})