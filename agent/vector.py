from langchain_chroma import Chroma

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PERSIST_DIR = "chroma_store/pdf"

# Load and split PDF
def create_pdf_vectorstore(pdf_path: str) -> Chroma:
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
    )

    return vectorstore

def load_vectorstore() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

    return vectorstore