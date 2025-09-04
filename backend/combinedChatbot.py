from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.docstore.document import Document

import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from dotenv import load_dotenv


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

load_dotenv()
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

start_url = "https://keydynamicssolutions.com/"
domain = "keydynamicssolutions.com"
max_pages = 100
delay = 1

visited = set()
to_visit = [start_url]
cleaned_data = []


def clean_text(html):
    """Extract and clean meaningful text from HTML."""
    soup = BeautifulSoup(html, "lxml")

    # Remove unwanted tags
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    # Extract text
    parts = []
    for tag in soup.find_all(["p", "h1", "h2", "h3", "li", "span"]):
        txt = tag.get_text(" ", strip=True)
        if txt:
            parts.append(txt)

    text = " ".join(parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def scrape_page(url):
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        print(f"Response from request : {response}")
    except Exception as e:
        print(f"\n\nScraping Failed at {url}: {e}")
        return None, []

    text = clean_text(response.text)

    # Extract links
    soup = BeautifulSoup(response.text, "lxml")
    links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]

    return text, links


def text_scraper():
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()

        if url in visited or domain not in url:
            continue

        print(f"Scraping URL : {url}")
        text, links = scrape_page(url)
        visited.add(url)

        if text:
            cleaned_data.append({"url": url, "content": text})

        for link in links:
            if link not in visited and domain in link:
                to_visit.append(link)

        time.sleep(delay)

    return cleaned_data

def init_rag():
    # Step 1: Crawl site
    print("Starting multi-page scraping...")
    scraped_data = text_scraper()
    print(f"✅ Scraped {len(scraped_data)} pages.")

    # Step 2: Convert to LangChain Documents
    docs = [
        Document(page_content=entry["content"], metadata={"source": entry["url"]})
        for entry in scraped_data if entry["content"].strip()
    ]
    # Step 3: Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(docs)

    # Step 4: Embeddings
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Step 5: Setup Qdrant
    qdrant_path = "./qdrant_data"
    os.makedirs(qdrant_path, exist_ok=True)
    client = QdrantClient(path=qdrant_path)

    collection_name = "kds_chatbot_collection"

    try:
        collections = client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections.collections)
        if not collection_exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    # Step 6: Insert docs
    try:
        qdrant = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings_model,
        )
        collection_info = client.get_collection(collection_name)
        if collection_info.points_count == 0:
            qdrant.add_documents(docs)
    except Exception:
        qdrant = QdrantVectorStore.from_documents(
            documents=docs,
            embedding=embeddings_model,
            client=client,
            collection_name=collection_name,
        )

    return qdrant


qdrant_store = init_rag()

def chat_node(state: ChatState):
    user_message = state['messages'][-1].content

    # Step 1: Retrieve context with similarity scores
    retrieved_docs_with_scores = qdrant_store.similarity_search_with_score(user_message, k=2)

    # Extract text + log scores
    context_parts = []
    print("\n🔎 Retrieved Documents & Similarity Scores:")
    for i, (doc, score) in enumerate(retrieved_docs_with_scores, 1):
        print(f"  {i}. Score: {score:.4f} | Source: {doc.metadata.get('source')}")
        print(f"     Snippet: {doc.page_content[:120]}...\n")  # preview text
        context_parts.append(doc.page_content)

    context_text = "\n\n".join(context_parts)

    # Step 2: Construct prompt
    prompt = f"""
You are a professional company spokesperson and specialized assistant. 
Your role is to represent the company in a clear, polite, and trustworthy manner while 
helping users with their queries. Always base your answers primarily on the retrieved context.

Context (retrieved information about the company, services, policies, and offerings):
{context_text}

User Query:
{user_message}

Instructions:
1. Use only the provided context to answer the user’s question. 
   - If the context contains the answer, explain it in a friendly and professional tone.
   - If the context only partially addresses the question, state what is known and politely clarify what is not available.
   - If the context does not have the information, say that you don’t have that detail and 
     suggest how the user can get help (e.g., contacting support, visiting a page, etc.).
2. Do not make up or assume facts beyond the given context.
3. Always write as if you are speaking on behalf of the company.
4. Keep responses concise, polite, and customer-focused.
5. If appropriate, provide step-by-step guidance or actionable next steps for the user.
"""

    response = llm.invoke(prompt)
    return {"messages": [HumanMessage(content=response)]}


def init_chatbot():
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)
    chatbot = graph.compile()
    return chatbot


chatbot = init_chatbot()


if __name__ == "__main__":
    while True:
        query = input("\nYour Question (type 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            break

        try:
            result = chatbot.invoke({"messages": [HumanMessage(content=query)]})
            print("\n=========================")
            print("User:", query)
            print("Bot:", result["messages"][-1].content)
            print("=========================\n")
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")
