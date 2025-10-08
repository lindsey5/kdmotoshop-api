from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from agent.config import get_model
from routes.predict_route import predict_router
from routes.ai_agent_route import agent_router
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kdmotoshop.onrender.com",  # your frontend URL
        "http://localhost:5173"             # local dev
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Mount routers
app.include_router(predict_router)
app.include_router(agent_router)

# Root route (for Render health check)
@app.get("/")
async def root():
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

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Bind to 0.0.0.0 so Render can detect the open port
    uvicorn.run("main:app", host="0.0.0.0", port=port)
