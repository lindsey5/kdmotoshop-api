from langchain.tools import tool
import os
from .config import qa_chain, db_chain

url = os.environ.get("URL")

@tool
def ask_question(question: str) -> str:
    """Use this to answer customer questions"""
    result = qa_chain({"query": question})
    return result["result"]

@tool
def products_tool(query: str) -> str:
    """Search, sort or filter for products from the KD Moto Shop inventory."""
    result = db_chain({ "query" : query })
    return result["result"]
    
    
tools = [products_tool, ask_question]