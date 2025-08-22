import os
import requests 
from langchain.tools import tool
from agent.db import get_products_collection
from .utils import _format_product
from .chain import create_rag_chain
from agent.vector import load_vectorstore

url = os.environ.get("URL")

# vectorstore = create_pdf_vectorstore("data/qa.pdf")
vectorstore = load_vectorstore()
qa_chain = create_rag_chain(vectorstore)


@tool
def ask_question(question: str) -> str:
    """Search for knowledge-based answers related to customer questions"""
    result = qa_chain({"query": question})
    return result["result"]


@tool
def products_tool() -> str:
    """Search, sort or filter products by (product name, stock, prices, rating) from the KD Moto Shop inventory."""
    products = list(get_products_collection().find({}))
    if not products:
        return "No products found."

    return [_format_product(product) for _, product in enumerate(products)]


@tool
def get_top_products() -> str:
    """Fetch the most selling products from the API."""
    try:
        response = requests.get(f"{url}/api/products/top?limit=100")
        response.raise_for_status()  # raise error if status != 200

        data = response.json()
        products = data.get("topProducts", [])

        if not products:
            return "No top products found."

        result = ""
        for p in products:
            result += (
                f'Product: {p["product_name"]}\n'
                f'Price: {p["price"]}\n'
                f'Quantity Sold: {p["totalQuantity"]}\n'
                f'Rating: {p["rating"]}\n\n'
            )
        return result.strip()

    except Exception as e:
        return f"Error fetching top products: {str(e)}"


def getTools():
    return [products_tool, ask_question, get_top_products]
