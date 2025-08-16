from agent.db import load_db
from agent.tools import getTools
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
load_db()
os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY")

memory = MemorySaver()
    
prompt = """
    You are a friendly and supportive chatbot assistant for KD MotoShop, your role is to answer customer questions related to KD Motoshop Website.  
    Guidelines:  
        - Answer questions about FAQs, Privacy Policies, Terms and Conditions
        - Maintain a polite and helpful tone at all times  
        - When presenting products, emphasize important features and display the information in a clear, easy-to-read format (Do not use asterisks)
"""

agent_executor = create_react_agent(
    model=init_chat_model("gemini-2.0-flash", model_provider="google_genai"), 
    tools=getTools(), 
    prompt=prompt,
    checkpointer=memory,
)