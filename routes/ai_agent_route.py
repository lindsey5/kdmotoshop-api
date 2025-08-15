from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from agent.agent import agent_executor
import uuid

agent_router = APIRouter()

@agent_router.post("/api/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        
        # Get thread_id from request or generate a new one
        thread_id = body.get("thread_id") or str(uuid.uuid4())
        user_message = body.get("message")
        
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

        return JSONResponse(content={"response": result, "success": True, "thread_id": thread_id})

    except Exception as e:
        print("Error in /api/chat:", str(e))
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)