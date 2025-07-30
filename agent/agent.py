from agent.db import load_db
from agent.tools import getTools
from .config import get_model
from langgraph.prebuilt import create_react_agent

from langgraph.checkpoint.memory import MemorySaver

load_db()

def get_agent_executor():
    memory = MemorySaver()
    
    prompt = """
        You are a friendly and supportive chatbot assistant for KD Moto Shop.  
        Your role is to answer customer questions about the website.
        Guidelines:  
        - Maintain a polite and helpful tone at all times  
        - When presenting products, emphasize important features and display the information in a clear, easy-to-read format
    """

    return create_react_agent(
        model=get_model(), 
        tools=getTools(), 
        prompt=prompt,
        checkpointer=memory,
    )