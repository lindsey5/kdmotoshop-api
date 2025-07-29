from langchain.tools import tool
import os
from agent.db import get_products_collection
from agent.utils import create_db_vectorstore, create_pdf_vectorstore, create_rag_chain

url = os.environ.get("URL")

docs = list(get_products_collection().find({}))

dbstore = create_db_vectorstore(docs=docs)
db_chain = create_rag_chain(dbstore)

vectorstore = create_pdf_vectorstore("data/qa.pdf")
qa_chain = create_rag_chain(vectorstore)

@tool
def ask_question(question: str) -> str:
    """Use this to answer customer questions"""
    result = qa_chain({"query": question})
    return result["result"]

@tool
def products_tool(query: str) -> str:
    """Search, sort or filter products from the KD Moto Shop inventory."""
    result = db_chain({ "query" : query })
    return result["result"]
    
    
tools = [products_tool, ask_question]