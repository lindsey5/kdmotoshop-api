from langchain.tools import tool
import os
from .config import qa_chain, db_chain

url = os.environ.get("URL")

@tool
def ask_question(question: str) -> str:
    """Ask a question"""
    result = qa_chain({"query": question})
    return result["result"]

@tool
def search_product(query: str) -> str:
    """Search or filter for products from the KD Moto Shop inventory."""
    result = db_chain({ "query" : query })
    return result["result"]
    
    
tools = [search_product, ask_question]