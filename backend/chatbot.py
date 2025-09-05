from fastapi import APIRouter
from pydantic import BaseModel
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

load_dotenv()

# Example LLM: can replace with ChatOllama / ChatGoogleGenerativeAI =======
llm = ChatOpenAI(model="gpt-4o-mini")

def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

checkpointer = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])

sessions = {}

class InitChatRequest(BaseModel):
    thread_id: str

class QueryRequest(BaseModel):
    thread_id: str
    query: str
    results: list[dict]


@router.post("/init")
def init_chat(req: InitChatRequest):
    """
    Initialize a chatbot session with memory persistence
    """
    sessions[req.thread_id] = chatbot
    return {"message": f"Chat session {req.thread_id} initialized."}


@router.post("/ask")
def chat_with_bot(req: QueryRequest):
    """
    Send user query + retrieved context to chatbot.
    """
    if req.thread_id not in sessions:
        return {"error": f"Session {req.thread_id} not initialized. Call /chatbot/init first."}

    bot = sessions[req.thread_id]

    context_texts = [r["content"] for r in req.results]
    context = "\n\n".join(context_texts)

    system_msg = SystemMessage(content=f"Use the following context to answer:\n{context}")
    user_msg = HumanMessage(content=req.query)

    # Invoke bot with memory persistence
    config = {"configurable": {"thread_id": req.thread_id}}
    response = bot.invoke({"messages": [system_msg, user_msg]}, config=config)

    return {"response": response["messages"][-1].content}
