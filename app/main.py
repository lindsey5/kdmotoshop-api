from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from agent.config import get_model
from routes.predict_route import predict_router   
from routes.ai_agent_route import agent_router
import uvicorn
import os

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kdmotoshop.onrender.com",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Mount FastAPI routers
app.include_router(predict_router)
app.include_router(agent_router)

@app.get("/")
async def run():
    try:
        response = get_model().invoke([{"role": "user", "content": "H"}])
        print(f"Agent raw response: {response}")

        if hasattr(response, "content"):
            return JSONResponse(content={"response": response.content})
        else:
            return JSONResponse(content={"response": str(response)})
    except Exception as e:
        print("Error in /:", str(e))
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
