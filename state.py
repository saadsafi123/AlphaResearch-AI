"""
State Schema: VigilAgent
Description: Defines the shared state dictionary (AgentState) used by LangGraph 
to pass context, memory, and execution variables between specialized agent nodes.
"""

from typing import Annotated, List, Literal, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """The global state object for the multi-agent workflow."""
    
    # Conversation History
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Clarity Agent Outputs
    clarity_status: Literal["clear", "needs_clarification"]
    
    # Research Agent Outputs
    current_company: Optional[str]
    raw_research_data: Optional[str]
    confidence_score: int  # Scale: 0 to 10
    
    # Validator Agent Outputs
    validation_result: Literal["sufficient", "insufficient"]
    research_attempts: int  # Tracks retry loops (Max threshold typically 3)