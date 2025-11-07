from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def create_pdf_vectorstore(pdf_path: str) -> Chroma:
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore


def load_vectorstore() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore
