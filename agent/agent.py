from agent.db import load_db
from agent.tools import getTools
from .config import get_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_db()

memory = MemorySaver()
    
prompt = """
    You're name is Echo a friendly and supportive chatbot assistant for KD MotoShop, your role is to answer customer questions related to KD Motoshop Website.  
    Guidelines:  
        - Answer questions about FAQs, Privacy Policies, Terms and Conditions
        - Maintain a polite and helpful tone at all times  
        - When presenting products, emphasize important features and display the information in a clear, easy-to-read format (Do not use asterisks)
"""

agent_executor = create_react_agent(
     model=get_model(), 
    tools=getTools(), 
    prompt=prompt,
    checkpointer=memory,
)