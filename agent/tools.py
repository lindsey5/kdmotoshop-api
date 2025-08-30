import os
from typing import List
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

import os
import requests
from langchain.tools import tool

# Environment variables
PAGE_ID = os.getenv("FB_PAGE_ID")
ACCESS_TOKEN = os.getenv("FB_ACCESS_TOKEN")

# Tool description
tool_prompt = (
    "Generate and post a marketing message with an image to a Facebook Page. "
    "Provide raw product details and the AI will create compelling marketing copy."
)

def create_post(caption: str, image_urls: List[str]) -> str:
    print(f"Caption: {caption}")
    print(f"Image URLs: {image_urls}")

    try:
        photo_ids = []
        # First, upload each image to get its ID
        for image_url in image_urls:
            upload_url = f"https://graph.facebook.com/{PAGE_ID}/photos"
            payload = {
                "url": image_url,
                "published": False,  # upload without posting
                "access_token": ACCESS_TOKEN,
            }
            response = requests.post(upload_url, data=payload)
            response.raise_for_status()
            photo_id = response.json().get("id")
            if photo_id:
                photo_ids.append({"media_fbid": photo_id})

        # Now create a single post with all uploaded images
        post_url = f"https://graph.facebook.com/{PAGE_ID}/feed"
        payload = {
            "message": caption,
            "attached_media": photo_ids,
            "access_token": ACCESS_TOKEN,
        }
        response = requests.post(post_url, json=payload, timeout=10000) 
        response.raise_for_status()
        return response.text

    except Exception as e:
        print("Error creating Facebook post:", e)
        return "Failed to create Facebook post."

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model

def generate_caption(product_details: str) -> str:
    model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    
    messages = [
        SystemMessage(
            content=(
                """Create one engaging and persuasive Facebook post caption 
based on the following product details. Make it lively, social-media-friendly, 
and include a clear call-to-action with the store details: 
Store Link: https://kdmotoshop.onrender.com/
Exact Address: Blk. 2 Lot 19 Phase 1 Brgy. Pinagsama, Taguig City
📍Search mo lang po sa Google Maps/Waze:
KD Motoshop Helmet Store Pinagsama Taguig
Landmark: near Phase 1 Arko (C5 Service Road)
🕘Store Hours
Open Daily: 9:00am-9:00pm
☎️Contact No.: 09931793845 / 09910735752
🛒Online Shop:
Shopee: https://ph.shp.ee/D2P7Bbe
Lazada: https://s.lazada.com.ph/s.tn8GB
Tiktok: https://vt.tiktok.com/ZSB3XN2Je/?page=TikTokShop"""
            )
        ),
        HumanMessage(content=product_details),
    ]

    response = model.invoke(messages)
    return response.content

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