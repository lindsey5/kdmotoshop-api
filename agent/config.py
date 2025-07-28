import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from agent.db import get_products_collection
from agent.utils import create_pdf_vectorstore, create_rag_chain, create_db_vectorstore

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY")

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

docs = list(get_products_collection().find({}))

dbstore = create_db_vectorstore(docs=docs)
db_chain = create_rag_chain(dbstore)

vectorstore = create_pdf_vectorstore("data/qa.pdf")
qa_chain = create_rag_chain(vectorstore)