from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv()

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Load and process documents
extracted_documents = load_pdf_files(directory= "data/")

# Filter documents to retain only 'source' in metadata and 'page_content'
minimal_docs = filter_to_minimal_docs(extracted_documents)

# Split the documents into smaller chunks for better processing
texts_chunk= text_split(minimal_docs)

# Download embeddings model
embeddings = download_embeddings()


# Initialize Pinecone
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key) 

# Create a Pinecone index if it doesn't exist

index_name = "healthcare-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of the embeddings
        metric="cosine",  # Similarity metric
        spec = ServerlessSpec(cloud= "aws", region= "us-east-1")
    )

index = pc.Index(index_name)  

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=index_name
)



