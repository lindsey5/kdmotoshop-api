from agent.db import load_db
from agent.tools import getChatbotTools, facebook_post_tool
from .config import get_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_db()

memory = MemorySaver()
    
chat_bot_prompt = """
    You're name is KD MotoBot a friendly and supportive chatbot assistant for KD MotoShop, your role is to answer customer questions related to KD Motoshop Website.  
    Guidelines:  
        - Answer questions about FAQs, Privacy Policies, Terms and Conditions
        - Maintain a polite and helpful tone at all times  
        - Always emphasize important features and display the information in html body content format, display image if available, and style it to make it presentable but dont put background
        - Always make the product name bold
        - Show only 5 products and always include a "Type 'See More' for more" prompt.
"""

chat_bot_agent = create_react_agent(
    model=get_model(), 
    tools=getChatbotTools(), 
    prompt=chat_bot_prompt,
    checkpointer=memory,
)

fb_ai_agent_prompt = """You're a ai agent that automate marketing posts"""

fb_ai_agent = create_react_agent(
    model=get_model(), 
    tools=[facebook_post_tool], 
    prompt=fb_ai_agent_prompt,
)