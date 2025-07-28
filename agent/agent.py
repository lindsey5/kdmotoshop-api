from .config import model
from .tools import tools
from langgraph.prebuilt import create_react_agent

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

prompt = """You are a helpful chatbot assistant for KD Moto Shop. Your role is to answer customer questions."""

agent_executor = create_react_agent(
    model, 
    tools, 
    prompt=prompt,
    checkpointer=memory,
)