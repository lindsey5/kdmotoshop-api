import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
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

# Create RAG chain
def create_qa_chain(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=GoogleGenerativeAI(model="models/gemini-2.0-flash", google_api_key=os.environ.get("GEMINI_API_KEY")),  
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain