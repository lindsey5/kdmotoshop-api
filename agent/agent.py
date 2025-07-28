from .config import model
from .tools import tools
from langgraph.prebuilt import create_react_agent

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

prompt = """You are a helpful chatbot assistant for KD Moto Shop. 
Your role is to answer customer questions.

Guidelines:
- Always be polite and helpful
- When showing products, highlight key details like price, stock, and category
- Always show product availability and pricing clearly
- If no products are found, suggest alternative search terms or categories
"""

agent_executor = create_react_agent(
    model, 
    tools, 
    prompt=prompt,
    checkpointer=memory,
)