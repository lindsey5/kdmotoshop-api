from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def create_pdf_vectorstore(pdf_path: str) -> Chroma:
    # Load PDF
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )

    return vectorstore

def load_vectorstore() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings 
    )

    return vectorstore
