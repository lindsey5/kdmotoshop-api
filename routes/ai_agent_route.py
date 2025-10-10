# routes/ai_agent_route.py

import re
import uuid
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from agent.agent import get_chat_bot_agent, get_fb_ai_agent

agent_router = APIRouter()

@agent_router.post("/api/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        thread_id = body.get("thread_id") or str(uuid.uuid4())
        user_message = body.get("message")

        input_message = {"role": "user", "content": user_message}
        config = {"configurable": {"thread_id": thread_id}}

        agent = get_chat_bot_agent()
        result = ""
        for step, metadata in agent.stream(
            {"messages": [input_message]},
            config=config,
            stream_mode="messages"
        ):
            if metadata["langgraph_node"] == "agent" and (text := step.text()):
                result += text

        cleaned = re.sub(r"```[a-zA-Z]*\n?", "", result).replace("```", "")
        return JSONResponse(content={"response": cleaned.strip(), "success": True, "thread_id": thread_id})

    except Exception as e:
        print("Error in /api/chat:", str(e))
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)

@agent_router.post("/api/generate_post")
async def generate_post(request: Request):
    try:
        body = await request.json()
        images = body.get("images")
        product_details = body.get("product_details")

        images_str = ", ".join(images) if isinstance(images, list) else images
        user_input = f"Create a Facebook post for this product: {product_details}. Use these images: {images_str}"

        config = {"configurable": {"thread_id": "123"}}
        agent = get_fb_ai_agent()

        result = ""
        for step, metadata in agent.stream(
            {"messages": [user_input]},
            config=config,
            stream_mode="messages"
        ):
            if metadata["langgraph_node"] == "agent" and (text := step.text()):
                result += text

        return JSONResponse(content={"response": result, "success": True})

    except Exception as e:
        print("Error in /api/generate_post:", str(e))
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)
