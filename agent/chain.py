from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAI
import os
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

# Create RAG chain
def create_rag_chain(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5 })
    qa_chain = RetrievalQA.from_chain_type(
        llm=GoogleGenerativeAI(model="models/gemini-2.0-flash", google_api_key=os.environ.get("GEMINI_API_KEY")),  
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        
    )
    return qa_chain