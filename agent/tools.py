from langchain.tools import tool
import requests
from typing import Optional, Dict, Any
import os
from .config import qa_chain

url = os.environ.get("URL")

def _format_product(product: Dict[str, Any], index: int) -> str:
    product_name = product.get('product_name', 'N/A')
    category = product.get('category', 'N/A')
    rating = product.get('rating', 0)
    variants = product.get('variants', [])

    if product.get('product_type') == 'Single':
      price = product.get('price', 0)
      stock = product.get('stock', 0)
      stock_text = f"{stock} units" if stock > 0 else "Out of stock"

    result = f"{index}: {product_name}\n"
    result += f"Category: {category}\n"

    if product.get('product_type') == 'Single':
      result += f"Price: ₱{price:.2f}\n"
      result += f"Stock: {stock_text}\n"
    result += f"Rating: ({rating}/5)\n"
    
    # Handle variants
    if variants:
        for j, variant in enumerate(variants, 1): 
            variant_price = variant.get('price', 0)
            variant_stock = variant.get('stock', 0)
            attributes = variant.get("attributes", {})
            attr_text = ", ".join(f"{value}" for key, value in attributes.items())
            result += f"{j}.{attr_text}:\n   Price: ₱{variant_price:.2f}, Stock: {variant_stock}\n"
    
    result += "\n"
    return result

@tool
def ask_question(question: str) -> str:
    """Ask a question based on the PDF"""
    result = qa_chain({"query": question})
    return result["result"]

@tool
def search_product_api(
    query: str = "", 
    page: int = 1, 
    limit: int = 10,
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
) -> str:
    """
    Search for products using the KD Moto Shop API.
    
    Args:
        query: Search term for products
        page: Page number (default: 1)
        limit: Products per page (default: 10)
        category: Filter by category
        min_price: Minimum price filter
        max_price: Maximum price filter

    Returns:
        Formatted string with product details
    """
    endpoint = f"{url}/api/product/reserved?visibility=Published"
    
    params = {
        "limit": limit,
        "page": page,
    }
    
    if query:
        params["searchTerm"] = query
    if category:
        params["category"] = category
    if min_price is not None:
        params["min"] = min_price
    if max_price is not None:
        params["max"] = max_price
    
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            products = data.get('products', data.get('data', []))
            total = data.get('total', data.get('totalCount', len(products) if products else 0))
            current_page = data.get('page', data.get('currentPage', page))
            total_pages = data.get('totalPages', data.get('pages', max(1, (total + limit - 1) // limit)))
            
            if not products:
                return "No products found"
            
            result = f"Found {total} products:\n\n"
            
            for i, product in enumerate(products, 1):
                result += _format_product(product, i)

            if total_pages > 1:
                result += f"\nPage {current_page} of {total_pages} (Total: {total} products)\n"

            return result
            
        else:
            return f"API failed with status {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"
    
def get_categories() -> str:
    """Use to get categories"""
    try:
        endpoint = f"{url}/api/category"
        response = requests.get(endpoint, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            categories = data.get('categories', data.get('data', []))

            if not categories:
                return "No categories found"
            
            result = "" 
            
            for i, category in enumerate(categories, 1):
                category_name = category.get('category_name')
                result += f"\n{category_name}"

            return result
            
        else:
            return f"API failed with status {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"
    
tools = [search_product_api, get_categories, ask_question]