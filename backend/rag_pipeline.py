from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os


def rag_pipeline(input_query):
    # Loading the documents
    print(f"\n\n\n Loading Docs... \n")
    loader = TextLoader("kds_website_data/website_text.txt")
    website_text = loader.load()
    print(website_text[0].page_content)

    # Splitting into chunks
    print(f"\n\n\n Splitting into chunks... \n")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(website_text)
    print(f" ---- \n\n\n Splitted into chunks \n\n\n {len(docs)}")

    # Embedding Model
    print(f"\n\n\n Creating Embedding Vector... \n")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Creating directory for Qdrant data
    qdrant_path = "./qdrant_data"
    os.makedirs(qdrant_path, exist_ok=True)
    
    print(f"\n\n\n Qdrant Client Store creation... \n")
    client = QdrantClient(path=qdrant_path)

    collection_name = "kds_chatbot_collection"
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections.collections)
        
        if not collection_exists:
            print("Creating new collection...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE), 
            )
            print("Collection created successfully!")
        else:
            print("Collection already exists!")
            
    except Exception as e:
        print(f"Error with collection operations: {e}")
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE), 
            )
            print("Collection created on retry!")
        except Exception as create_error:
            print(f"Failed to create collection: {create_error}")

    try:
        qdrant = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings_model,
        )
        
        collection_info = client.get_collection(collection_name)
        if collection_info.points_count == 0:
            print("Collection is empty. Adding documents...")
            # Add documents to empty collection
            qdrant.add_documents(docs)
            print("Documents added to collection!")
        else:
            print(f"Collection already has {collection_info.points_count} documents!")
            
    except Exception as e:
        print(f"Error with existing vector store, creating new one: {e}")
        qdrant = QdrantVectorStore.from_documents(
            documents=docs,
            embedding=embeddings_model,
            client=client,
            collection_name=collection_name,
        )

    print("Vector Store initialized successfully! \n\n")
    
    results = qdrant.similarity_search(input_query, k=2)

    print(f"Found {len(results)} similar documents:")
    for i, res in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Content: {res.page_content}")
        print(f"Metadata: {res.metadata}")


if __name__ == "__main__":
    rag_pipeline("What are services by key dynamics solutions?")