from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from agent.agent import get_agent_executor
import uuid

agent_bp = Blueprint("agent", __name__)

@agent_bp.route("/api/chat", methods=['POST', 'OPTIONS'])
@cross_origin(
    origins=["https://kdmotoshop.onrender.com", "http://localhost:5173"],
    supports_credentials=True,
    methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)
def chat():
    try:
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
        for step, metadata in get_agent_executor().stream(
            {"messages": [input_message]},
            config=config,
            stream_mode="messages"
        ):
            if metadata["langgraph_node"] == "agent" and (text := step.text()):
                result += text

        return jsonify({"response": result, "success": True, "thread_id" : thread_id })

    except Exception as e:
        print("Error in /api/chat:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500