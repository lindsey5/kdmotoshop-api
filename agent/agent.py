from .config import model
from .tools import tools
from langgraph.prebuilt import create_react_agent

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

prompt = """
    You are a friendly and supportive chatbot assistant for KD Moto Shop.  
    Guidelines:  
    - Maintain a polite and helpful tone at all times  
    - When presenting products, emphasize important features and display the information in a clear, easy-to-read format
"""

agent_executor = create_react_agent(
    model, 
    tools, 
    prompt=prompt,
    checkpointer=memory,
)