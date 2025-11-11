from agent.config import get_model
from agent.db import load_db
from agent.tools import getChatbotTools, facebook_post_tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Globals
_model = None
_memory = None
_chat_bot_agent = None
_fb_ai_agent = None

def initialize_agents():
    """
    Initialize model and agents. Safe to call multiple times.
    """
    global _model, _memory, _chat_bot_agent, _fb_ai_agent

    if _model is None:
        print("Loading model...")
        _model = get_model()

    if _memory is None:
        _memory = MemorySaver()

    if _chat_bot_agent is None:
        print("Loading DB for chat bot...")
        load_db()
        chat_bot_prompt = """
            You're name is KD MotoBot, a friendly and supportive chatbot assistant for KD MotoShop. 
            Guidelines:  
            - Always display the information in html body content format, display image if available put it on <img /> tag, and style it to make it presentable but dont put background
            - Answer FAQs, Privacy Policies, Terms, etc.
            - Maintain polite and helpful tone
        """
        _chat_bot_agent = create_react_agent(
            model=_model,
            tools=getChatbotTools(),
            prompt=chat_bot_prompt,
            checkpointer=_memory
        )

    if _fb_ai_agent is None:
        fb_ai_agent_prompt = "You're an AI agent that automates marketing posts."
        _fb_ai_agent = create_react_agent(
            model=_model,
            tools=[facebook_post_tool],
            prompt=fb_ai_agent_prompt
        )

def get_chat_bot_agent():
    """Getter for chat bot agent"""
    return _chat_bot_agent

def get_fb_ai_agent():
    """Getter for Facebook AI agent"""
    return _fb_ai_agent

def get_model_instance():
    """Getter for model"""
    return _model