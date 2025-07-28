import os
from typing import Any, Dict, List
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA

# Load and split PDF
def create_pdf_vectorstore(pdf_path: str) -> FAISS:
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

def create_db_vectorstore(docs: List[Dict[str, Any]]) -> Chroma:
    texts = [_format_product(doc) for _, doc in enumerate(docs)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs_split = splitter.create_documents(texts)
    # Embed and store in Chroma (in-memory or persistent)
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    vectorstore = Chroma.from_documents(docs_split, embedding)

    return vectorstore

# Create RAG chain
def create_rag_chain(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=GoogleGenerativeAI(model="models/gemini-2.0-flash", google_api_key=os.environ.get("GEMINI_API_KEY")),  
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain


def _format_product(product: Dict[str, Any]) -> str:
    product_name = product.get('product_name', 'N/A')
    category = product.get('category', 'N/A')
    rating = product.get('rating', 0)
    variants = product.get('variants', [])

    if product.get('product_type') == 'Single':
      price = product.get('price', 0)
      stock = product.get('stock', 0)
      stock_text = f"{stock} units" if stock > 0 else "Out of stock"

    result = f"{product_name}\n"
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