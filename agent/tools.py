import os
from typing import List
import requests 
from langchain.tools import tool
from agent.db import get_products_collection
from .utils import _format_product, create_post, generate_caption
from .chain import create_rag_chain
from agent.vector import load_vectorstore
import os
import requests
from langchain.tools import tool

url = os.environ.get("URL")

# vectorstore = create_pdf_vectorstore("data/qa.pdf")
vectorstore = load_vectorstore()
qa_chain = create_rag_chain(vectorstore)

@tool
def ask_question(question: str) -> str:
    """Search for knowledge-based answers related to customer questions"""
    try:
        result = qa_chain({"query": question})
        print(result)
        return result["result"]
    except Exception as e:
            return f"{str(e)}"


@tool
def products_tool() -> str:
    """Search, sort, or filter products by (product name, stock, prices, rating) 
    from the KD Moto Shop inventory"""
    products = list(get_products_collection().find({}))
    if not products:
        return "No products found."

    return [_format_product(product) for _, product in enumerate(products)]


@tool
def get_top_products() -> str:
    """Fetch the most selling products from the API"""
    try:
        response = requests.get(f"{url}/api/products/top")
        response.raise_for_status() 

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
                f'Image: {p["image"]}\n\n'
            )

        return result.strip()

    except Exception as e:
        return f"Error fetching top products: {str(e)}"

def getChatbotTools():
    return [products_tool, ask_question, get_top_products]

@tool
def facebook_post_tool(product_details: str, images: List[str]) -> str:
    """
    Automatically generate an AI marketing caption from product details 
    and publish it with the product image to a Facebook Page.
    
    Args:
        product_details: Description or name of the product to promote.
        images: Publicly accessible URLs of the product image.
    
    Returns:
        The result of the Facebook post creation (e.g., post ID or confirmation message).
    """
    # Generate caption dynamically using Gemini AI
    caption = generate_caption(product_details)

    # Post to Facebook
    return create_post(caption, images)