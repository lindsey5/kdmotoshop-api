from langchain.tools import tool
import os

from agent.db import get_products_collection
from .utils import _format_product, create_rag_chain
from agent.vector import create_pdf_vectorstore, load_vectorstore

url = os.environ.get("URL")

# vectorstore = create_pdf_vectorstore("data/qa.pdf")
vectorstore = load_vectorstore()
qa_chain =  create_rag_chain(vectorstore)

@tool
def ask_question(question: str) -> str:
    """Find answers related to customer question"""
    result = qa_chain({"query": question})
    return result["result"]

@tool
def products_tool() -> str:
    """Search, sort or filter products by (product name, stock, prices, rating) from the KD Moto Shop inventory."""
    products = list(get_products_collection().find({}))
    if not products:
        return "No products found."
    
    return [_format_product(product) for _, product in enumerate(products)]

def getTools():
    tools = [products_tool, ask_question]
    return tools