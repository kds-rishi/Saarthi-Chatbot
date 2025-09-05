from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from langchain.chat_models import GoogleGemini
from langchain.schema import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    response: str


load_dotenv()


try:
    llm = GoogleGemini(model="gemini-2.5-flash")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm = None


# Creating Node
def chat_node(state: ChatState):
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized")
   
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


def init_chatbot():
    checkpointer = MemorySaver()
    graph = StateGraph(ChatState)
   
    graph.add_edge(START, 'chat_node')
    graph.add_node('chat_node', chat_node)
    graph.add_edge('chat_node', END)
   
    chatbot = graph.compile(checkpointer=checkpointer)
    return chatbot


@app.post("/api/chatbot", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Create thread config for memory
        thread_config = {"configurable": {"thread_id": request.user_id}}
       
        # Create the human message
        human_message = HumanMessage(content=request.message)
       
        # Invoke the chatbot with the message and thread config
        result = chatbot.invoke(
            {"messages": [human_message]},
            config=thread_config
        )
       
        # Extract the response message
        response_message = result["messages"][-1]
       
        return ChatResponse(response=response_message.content)
   
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


