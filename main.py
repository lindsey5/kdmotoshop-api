# main.py

import asyncio
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routes.predict_route import predict_router
from routes.ai_agent_route import agent_router
from agent.agent import get_model_instance
import uvicorn

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kdmotoshop.onrender.com",
        "https://kdmotoshop-bc7u.onrender.com",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(initialize_agents_async())  # run in background

async def initialize_agents_async():
    from agent.agent import initialize_agents
    initialize_agents()

# Mount routers
app.include_router(predict_router)
app.include_router(agent_router)

@app.get("/")
async def root():
    try:
        model = get_model_instance()
        response = model.invoke([{"role": "user", "content": "H"}])
        return JSONResponse(content={"response": getattr(response, "content", str(response))})
    except Exception as e:
        
        print("Error in /:", str(e))
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
