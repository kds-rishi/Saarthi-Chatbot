from fastapi import APIRouter, Query
from pydantic import BaseModel
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings


class QueryRequest(BaseModel):
    query: str

router = APIRouter()

@router.post("/user-input")
def query_docs(request: QueryRequest):
    """
    Takes a user query and returns top 3 relevant chunks from vector DB.
    """
    client = QdrantClient(path="./qdrant_data")
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        qdrant = QdrantVectorStore(
            client=client,
            collection_name="kds_chatbot_collection",
            embedding=embeddings_model,
        )

        results = qdrant.similarity_search(request.query, k=3)

        return {
            "query": request.query,
            "results": [
                {"content": res.page_content, "metadata": res.metadata}
                for res in results
            ]
        }

    except Exception as e:
        return {"error": str(e)}

