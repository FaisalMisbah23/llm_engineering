from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from litellm import completion
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv(override=True)

MODEL = "groq/openai/gpt-oss-20b"
DB_NAME = "preprocessed_db"
collection_name = "docs"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
KNOWLEDGE_BASE_PATH = Path("knowledge-base")
AVERAGE_CHUNK_SIZE = 500

openai = OpenAI()

# Result class for search results
class Result(BaseModel):
    page_content: str
    metadata: dict

# Chunk class representing a document chunk
class Chunk(BaseModel):
    headline: str = Field(description="Brief heading for this chunk")
    summary: str = Field(description="Summary of the chunk content")
    content: str = Field(description="Full content of the chunk")
    token_count: int = Field(description="Number of tokens in chunk")

# Database operations
class VectorDatabase:
    def __init__(self, db_name=DB_NAME, collection_name=collection_name):
        self.client = PersistentClient(path=db_name)
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def add_document(self, content, metadata):
        embedding = embedding_model.embed_query(content)
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[metadata.get("id", str(hash(content)))]
        )
    
    def search(self, query, n_results=5):
        query_embedding = embedding_model.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

# Main RAG knowledge worker
def answer_question(question, db):
    # Search for relevant documents
    search_results = db.search(question, n_results=5)
    
    # Build context from results
    context = "\n\n".join([
        doc for doc in search_results.get("documents", [[]])[0]
    ])
    
    # Create prompt with context
    prompt = f"""
    You are an AI knowledge worker with access to the following documents:
    
    {context}
    
    Question: {question}
    
    Please provide a comprehensive answer based solely on the document content.
    """
    
    # Get answer from LLM
    response = completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful AI knowledge worker."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    db = VectorDatabase()
    print("RAG Knowledge Worker initialized. Ready for questions.")
