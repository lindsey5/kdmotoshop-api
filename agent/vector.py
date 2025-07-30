import os
from typing import Any, Dict, List
from langchain_chroma import Chroma

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.utils import _format_product

# Load and split PDF
def create_pdf_vectorstore(pdf_path: str) -> Chroma:
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    docs = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
    )

    return vectorstore

def create_db_vectorstore(docs: List[Dict[str, Any]]) -> Chroma:
    texts = [_format_product(doc) for _, doc in enumerate(docs)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    docs_split = splitter.create_documents(texts)
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    vectorstore = Chroma.from_documents(
        docs_split,
        embedding,
    )

    return vectorstore
