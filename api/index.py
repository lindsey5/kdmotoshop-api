from app.main import app
from mangum import Mangum  # ASGI adapter

handler = Mangum(app)      # Wrap FastAPI app for serverless