from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, SystemMessage
import requests
from langchain.chat_models import init_chat_model
import os

def _format_product(product: Dict[str, Any]) -> str:
    product_name = product.get('product_name', 'N/A')
    category = product.get('category', 'N/A')
    rating = product.get('rating', 0)
    variants = product.get('variants', [])
    thumbnail = product.get('thumbnail');

    if product.get('product_type') == 'Single':
      price = product.get('price', 0)
      stock = product.get('stock', 0)
      stock_text = f"{stock} units" if stock > 0 else "Out of stock"

    result = f"Product{product_name}\n"
    result += f"-Product image: {thumbnail.get('imageUrl')}\n"
    result += f"-Category: {category}\n"

    if product.get('product_type') == 'Single':
      result += f"-Price: ₱{price:.2f}\n"
      result += f"-Stock: {stock_text}\n"
      result += f"-Rating: ({rating}/5)\n"
      result += "Variants:\n"
    # Handle variants
    if variants:
        for j, variant in enumerate(variants, 1): 
            variant_price = variant.get('price', 0)
            variant_stock = variant.get('stock', 0)
            attributes = variant.get("attributes", {})
            
            attr_text = " | ".join(f"{value}" for key, value in attributes.items())
            result += f" ●{attr_text}\n    -Price: ₱{variant_price:.2f}, Stock: {variant_stock}\n"
    
    result += "\n\n\n"
    return result

# Environment variables
PAGE_ID = os.getenv("FB_PAGE_ID")
ACCESS_TOKEN = os.getenv("FB_ACCESS_TOKEN")

def getPageAccessToken():
    try:
        url = f"https://graph.facebook.com/v23.0/{PAGE_ID}"
        params = {
            "fields": "access_token",
            "access_token": ACCESS_TOKEN  # this should be your long-lived USER token
        }
        
        response = requests.get(url, params=params, timeout=10000)
        response.raise_for_status()  # raise error if response is not 200

        data = response.json()
        page_access_token = data.get("access_token")

        if not page_access_token:
            print("No access token found in response:", data)
            return None

        return page_access_token

    except Exception as e:
        print("Error creating page access token:", e)
        return None
    
def create_post(caption: str, image_urls: List[str]) -> str:
    print(f"Caption: {caption}")
    print(f"Image URLs: {image_urls}")

    try:
        page_token = getPageAccessToken()  # always use page access token
        if not page_token:
            return "Failed to get Page Access Token."

        photo_ids = []
        # First, upload each image to get its ID
        for image_url in image_urls:
            upload_url = f"https://graph.facebook.com/{PAGE_ID}/photos"
            payload = {
                "url": image_url,
                "published": False,  # upload without posting
                "access_token": page_token,
            }
            response = requests.post(upload_url, data=payload, timeout=10)
            response.raise_for_status()

            data = response.json()
            photo_id = data.get("id")
            if not photo_id:
                print("Upload failed:", data)
                return "Failed to upload image."

            photo_ids.append({"media_fbid": photo_id})

        # Now create a single post with all uploaded images
        post_url = f"https://graph.facebook.com/{PAGE_ID}/feed"
        payload = {
            "message": caption,
            "attached_media": photo_ids,
            "access_token": page_token,
        }
        response = requests.post(post_url, json=payload, timeout=10000)
        response.raise_for_status()
        return response.text

    except Exception as e:
        print("Error creating Facebook post:", e)
        return "Failed to create Facebook post."


def generate_caption(product_details: str) -> str:
    model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    
    messages = [
        SystemMessage(
            content=(
                """Create one tagalog engaging and persuasive Facebook post caption 
based on the following product details. Make it lively, social-media-friendly, 
and include a clear call-to-action with the store details (Highlight the store link): 
Exact Address: Blk. 2 Lot 19 Phase 1 Brgy. Pinagsama, Taguig City
📍Search mo lang po sa Google Maps/Waze:
KD Motoshop Helmet Store Pinagsama Taguig Landmark: near Phase 1 Arko (C5 Service Road)
🕘Store Hours
Open Daily: 9:00am-9:00pm
☎️Contact No.: 09931793845 / 09910735752
Visit: https://kdmotoshop.onrender.com for 10% discount and free shipping"""
            )
        ),
        HumanMessage(content=product_details),
    ]

    response = model.invoke(messages)
    return response.content