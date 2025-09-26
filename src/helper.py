# Import necessary modules

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate


# Extract text from PDF files in the 'data' directory

def load_pdf_files(directory):
    loader = DirectoryLoader(
        directory, 
        glob= "*.pdf",
        loader_cls= PyPDFLoader
    )
    documents = loader.load()
    return documents

# Filter documents to retain only 'source' in metadata and 'page_content'
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list containing only the 'source' in metadata and 'page_content'.
    """

    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata = {"source": src}
            )
        )

    return minimal_docs

# Split the documents into smaller chunks for better processing
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20,
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)

    return texts_chunk


def download_embeddings():
    """
    Download and return HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

