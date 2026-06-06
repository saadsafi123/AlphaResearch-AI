"""
Agent: Validator (QA Auditor)
Description: Acts as the quality control circuit breaker. Compares the user's 
query against the data retrieved by the Research Agent to ensure completeness 
before allowing the graph to proceed to Synthesis.
"""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from state import AgentState

class ValidationDecision(BaseModel):
    """Structured output schema for the Validator Agent's decision."""
    reasoning: str = Field(
        description="Brief explanation of why the data is sufficient or insufficient for the specific topic."
    )
    is_sufficient: str = Field(
        description="Respond 'yes' if the research answers the query completely. Respond 'no' if it is missing core context requested by the user."
    )

VALIDATOR_PROMPT = """You are an elite QA Auditor for an autonomous intelligence platform. 
Compare the User's Query against the Gathered Research Data.
Determine if the data contains enough specific facts, context, or data points to comprehensively answer the query.
If critical context, core explanations, or requested details are missing, mark it insufficient."""

def validator_agent(state: AgentState):
    """
    Evaluates the research data against the latest user prompt.
    Returns 'sufficient' or 'insufficient' to guide conditional routing.
    """
    user_query = state["messages"][-1].content if state["messages"] else "Unknown"
    research_data = state.get("raw_research_data", "")
    
    # Use temperature 0 for strict, deterministic logical evaluation
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(ValidationDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", VALIDATOR_PROMPT),
        ("user", "User Query: {user_query}\n\nGathered Data: {research_data}")
    ])
    
    chain = prompt | structured_llm
    decision = chain.invoke({"user_query": user_query, "research_data": research_data})
    
    status = "sufficient" if decision.is_sufficient.lower() == "yes" else "insufficient"
    
    print(f"\n[NODE EXECUTION] 🛡️ Validator Agent active. Result: {status.upper()}")
    print(f"   ↳ Reasoning: {decision.reasoning}") 
    
    return {"validation_result": status}