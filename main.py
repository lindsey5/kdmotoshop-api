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

allowed_origins = [
    "https://kdmotoshop.onrender.com",
    "http://localhost:5173",
]

print(allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

@app.middleware("http")
async def log_request(request, call_next):
    print(">>> Origin:", request.headers.get("origin"))
    response = await call_next(request)
    print("<<< Access-Control-Allow-Origin:", response.headers.get("access-control-allow-origin"))
    return response

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
