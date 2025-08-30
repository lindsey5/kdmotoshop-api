import re
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from agent.agent import chat_bot_agent, fb_ai_agent
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
        for step, metadata in chat_bot_agent.stream(
            {"messages": [input_message]},
            config=config,
            stream_mode="messages"
        ):
            if metadata["langgraph_node"] == "agent" and (text := step.text()):
                result += text

        cleaned = re.sub(r"```[a-zA-Z]*\n?", "", result)
        cleaned = cleaned.replace("```", "")
        
        return JSONResponse(content={"response": cleaned.strip(), "success": True, "thread_id": thread_id})

    except Exception as e:
        print("Error in /api/chat:", str(e))
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)
    
@agent_router.post("/api/generate_post")
async def generate_post(request: Request):
    try:
        body = await request.json()

        images = body.get("images")  # could be a list or a single string
        product_details = body.get("product_details")

        # If images is a list, join them into a string
        if isinstance(images, list):
            images_str = ", ".join(images)
        else:
            images_str = images  # single image

        print(images)

        # Pass input as a string
        user_input = f"Create a Facebook post for this product: {product_details}. Use these images: {images_str}"

        
        input_message = {"role": "user", "content": user_input}
        config = {
            "configurable": {
                "thread_id": "123"
            }
        }

        result = ""
        for step, metadata in fb_ai_agent.stream(
            {"messages": [input_message]},
            config=config,
            stream_mode="messages"
        ):
            if metadata["langgraph_node"] == "agent" and (text := step.text()):
                result += text

        return JSONResponse(content={"response": result, "success": True})

    except Exception as e:
        print("Error in /api/generate_post:", str(e))
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)