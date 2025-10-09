from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# Load and split PDF, then create a vectorstore
def create_pdf_vectorstore(pdf_path: str) -> Chroma:
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # ✅ Use local SentenceTransformer model (no API needed)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore


# Load an existing vectorstore from disk
def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore
