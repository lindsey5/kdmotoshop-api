from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
from routes.predict_route import predict_router   
from routes.ai_agent_route import agent_router
import uvicorn
import os

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Mount FastAPI routers
app.include_router(predict_router)
app.include_router(agent_router)

@app.get("/")
async def run():
    return JSONResponse(content={"response": "Hi."})

handler = Mangum(app)