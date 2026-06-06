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
        description="Brief explanation of why the data is sufficient or insufficient."
    )
    # Utilizing a string ('yes'/'no') instead of a boolean to prevent JSON parsing errors in LLM outputs
    is_sufficient: str = Field(
        description="Respond 'yes' if the research answers the query completely. Respond 'no' if it is missing key information requested by the user."
    )

VALIDATOR_PROMPT = """You are a rigorous QA Auditor for a research firm. 
Compare the User's Query against the Gathered Research Data.
Determine if the data contains enough specific facts to confidently answer the user's question.
If critical numbers, dates, or comparisons are missing, mark it insufficient."""

def validator_agent(state: AgentState):
    """
    Evaluates the research data against the latest user prompt.
    Returns 'sufficient' or 'insufficient' to guide conditional routing.
    """
    # Extract the latest user query for context
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
    
    # Map the string response to the system's execution flags
    status = "sufficient" if decision.is_sufficient.lower() == "yes" else "insufficient"
    
    # Print the execution status and the AI's reasoning to the terminal
    print(f"\n[NODE EXECUTION] 🛡️ Validator Agent active. Result: {status.upper()}")
    print(f"   ↳ Reasoning: {decision.reasoning}") 
    
    return {"validation_result": status}