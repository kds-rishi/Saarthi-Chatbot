import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain.docstore.document import Document

def store_embeddings():
    
    csv_path = "cleaned_scraped_data.csv"

    # Step 1: Ensure CSV exists
    if not os.path.exists(csv_path):
        return {
            "success": False,
            "error": f"CSV not found at {csv_path}",
            "stored_count": 0,
            "total_count": 0
        }
    

    # Step 2: Load CSV
    print(f"\n\n Loading Docs from CSV... \n")
    df = pd.read_csv(csv_path)

    if "content" not in df.columns:
        raise ValueError("CSV file must contain a 'content' column")

    website_texts = df["content"].dropna().tolist()
    print(f"Loaded {len(website_texts)} rows of text")

    documents = [Document(page_content=text) for text in website_texts]

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks")

    # Embeddings
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Qdrant setup
    qdrant_path = "./qdrant_data"
    os.makedirs(qdrant_path, exist_ok=True)
    client = QdrantClient(path=qdrant_path)

    collection_name = "kds_chatbot_collection"
    success, stored_count, total_count = False, 0, 0

    try:
        collections = client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections.collections)
        
        if not collection_exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            print("Collection created successfully!")
        else:
            print("Collection already exists.")
    except Exception as e:
        return {"success": False, "error": str(e), "stored_count": 0, "total_count": 0}

    # Use QdrantVectorStore (new API)
    try:
        qdrant = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings_model,
        )
        collection_info = client.get_collection(collection_name)
        total_count = collection_info.points_count

        if total_count == 0:
            print("Collection empty, adding documents...")
            qdrant.add_documents(docs)
            success = True
            stored_count = len(docs)
            total_count = client.get_collection(collection_name).points_count
        else:
            print(f"Collection already has {total_count} documents.")
            success = True
            stored_count = 0
    except Exception as err:
        return {"success": False, "error": str(err), "stored_count": 0, "total_count": 0}

    return {
        "success": success,
        "stored_count": stored_count,
        "total_count": total_count
    }
