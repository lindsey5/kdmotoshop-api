import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from agent.utils import create_pdf_vectorstore, create_qa_chain

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY")

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

vectorstore = create_pdf_vectorstore("data/qa.pdf")
qa_chain = create_qa_chain(vectorstore)