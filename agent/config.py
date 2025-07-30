import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from agent.db import load_db

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY")

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

load_db()