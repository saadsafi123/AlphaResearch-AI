"""
Graph Orchestration: AlphaResearch AI Workflow
Description: Defines the StateGraph logic, nodes, and conditional routing edges.
This file wires the 4 specialized agents together into a cyclic, memory-backed pipeline.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import State Schema and Specialized Agent Nodes
from state import AgentState
from agents.clarity import clarity_agent
from agents.research import research_agent
from agents.validator import validator_agent
from agents.synthesis import synthesis_agent

# --- Conditional Routing Logic ---

def route_clarity(state: AgentState):
    """Routes based on the Clarity Agent's ambiguity evaluation."""
    if state.get("clarity_status") == "needs_clarification":
        # Human-in-the-Loop (HITL): Pause execution and return to user
        return END 
    return "research_agent"

def route_research(state: AgentState):
    """Routes based on the Research Agent's self-evaluated confidence score."""
    score = state.get("confidence_score", 0)
    print(f"[ROUTING] Research Confidence Score is {score}/10")
    
    if score < 6:
        print("[ROUTING] 🧭 Diverting to Validator Agent for thorough audit.")
        return "validator_agent"
        
    print("[ROUTING] 🧭 Confidence high. Proceeding directly to Synthesis Agent.")
    return "synthesis_agent"

def route_validator(state: AgentState):
    """Routes based on QA validation result, enforcing a max retry limit."""
    attempts = state.get("research_attempts", 0)
    result = state.get("validation_result")
    
    print(f"[ROUTING] Validator evaluation: {result.upper()} | Total attempts so far: {attempts}")
    
    if result == "insufficient" and attempts < 3:
        # Cyclic routing: Force the research agent to try again
        print("🔄 [LOOP TRIGGERED] Data inadequate! Routing back to Research Agent for deeper web scrape.")
        return "research_agent"
        
    # Exit loop if data is sufficient OR max attempt threshold (3) is reached
    print("➡️ [ROUTE COMPLETED] Advancing to Synthesis Engine.")
    return "synthesis_agent"


# --- Build and Compile the Graph ---

builder = StateGraph(AgentState)

# 1. Add Nodes (The Workers)
builder.add_node("clarity_agent", clarity_agent)
builder.add_node("research_agent", research_agent)
builder.add_node("validator_agent", validator_agent)
builder.add_node("synthesis_agent", synthesis_agent)

# 2. Define Edges (The Flow)
builder.add_edge(START, "clarity_agent")
builder.add_conditional_edges("clarity_agent", route_clarity)
builder.add_conditional_edges("research_agent", route_research)
builder.add_conditional_edges("validator_agent", route_validator)
builder.add_edge("synthesis_agent", END)

# 3. Compile Graph with Memory (Checkpointer)
# MemorySaver ensures multi-turn conversation context is persisted across API calls
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)