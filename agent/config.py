import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables from .env if running locally
load_dotenv()

# Read GEMINI_API_KEY safely
gemini_key = os.getenv("GEMINI_API_KEY")

# Set GOOGLE_API_KEY only if available
if gemini_key:
    os.environ["GOOGLE_API_KEY"] = gemini_key
else:
    print("GEMINI_API_KEY not found! Make sure it's set in Render environment variables.")

def get_model():
    return init_chat_model("gemini-2.0-flash", model_provider="google_genai")
