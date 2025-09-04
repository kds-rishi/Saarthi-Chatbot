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


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


load_dotenv()
llm = GoogleGemini(model="gemini-2.5-flash")


# Creating Node 
def chat_node(state: ChatState):
    # take user query from state
    messages = state['messages']
    # send to llm
    response = llm.invoke(messages)
    # store the response in state - we send to list add_messages will add
    return {'messages' : [response]}


def init_chatbot():
    checkpointer = MemorySaver()

    graph = StateGraph(ChatState)

    # adding nodes
    graph.add_node('chat_node', chat_node)

    graph.add_edge(START, 'chat_node')
    graph.add_edge('chat_node', END)

    chatbot = graph.compile(checkpointer=checkpointer)