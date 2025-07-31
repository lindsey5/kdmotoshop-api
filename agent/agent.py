from agent.db import load_db
from agent.tools import getTools
from .config import get_model
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_tool_calling_agent, AgentExecutor

from langgraph.checkpoint.memory import MemorySaver

load_db()

memory = MemorySaver()
    
prompt = """
    You are a friendly and supportive chatbot assistant for KD Moto Shop.  
    Your role is to answer customer questions about the website.
    Guidelines:  
        - Maintain a polite and helpful tone at all times  
        - When presenting products, emphasize important features and display the information in a clear, easy-to-read format avoid asterisk
"""


agent = create_tool_calling_agent(get_model(), getTools(), prompt)
agent_executor = AgentExecutor(agent=agent, tools=getTools(), memory=memory, verbose=True)