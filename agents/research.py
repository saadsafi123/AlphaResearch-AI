"""
Agent: Research Engine
Description: Interfaces with the Tavily search tool to retrieve live, factual data. 
Evaluates the search context and scores its own confidence to inform the Validator.
"""

from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from state import AgentState

class ResearchEvaluation(BaseModel):
    """Structured output schema for the Research Agent."""
    findings: str = Field(description="Detailed compilation of news and financials.")
    confidence_score: int = Field(description="Score from 0 to 10 rating completeness.")

RESEARCH_PROMPT = """You are an expert equity researcher.
Target Subject: {company_name}
Analyze the search context below. Extract granular facts.
Rate your confidence (0-10) based on how well this answers the user's implicit needs."""

def research_agent(state: AgentState):
    """
    Executes a web search based on the extracted target entity, processes the 
    raw search results, and attempts to extract meaningful findings.
    """
    company = state.get("current_company")
    user_query = state["messages"][-1].content if state["messages"] else ""
    
    # Construct a robust search query utilizing context fallbacks
    search_query = f"{company or 'business'} {user_query} financials recent news".strip()
    
    search_tool = TavilySearchResults(max_results=5)
    
    # Safely invoke the external search API
    try:
        raw_results = search_tool.invoke({"query": search_query})
    except Exception as e:
        print(f"Tavily Tool Error: {e}")
        raw_results = f"Search failed due to an error: {str(e)}"

    # Gracefully handle string or unexpected API formats to prevent indexing crashes
    if isinstance(raw_results, str):
        search_context = f"Raw search context info: {raw_results}"
    elif isinstance(raw_results, list) and len(raw_results) > 0 and isinstance(raw_results[0], str):
        search_context = "\n\n".join([f"Result: {text}" for text in raw_results])
    elif isinstance(raw_results, list) and len(raw_results) > 0 and isinstance(raw_results[0], dict):
        search_context = "\n\n".join([f"Source: {r.get('url', 'N/A')}\nContent: {r.get('content', '')}" for r in raw_results])
    else:
        search_context = "No accessible research context could be retrieved for this query."

    # Initialize the reasoning model for data extraction
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2) 
    structured_llm = llm.with_structured_output(ResearchEvaluation)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESEARCH_PROMPT),
        ("user", "Raw Search Data:\n{context}\nGenerate findings and confidence score.")
    ])
    
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({"company_name": company or "Requested Subject", "context": search_context})
        findings = result.findings
        score = result.confidence_score
    except Exception as e:
        print(f"LLM Parsing Error: {e}")
        findings = f"We encountered an architectural error parsing information for this query. Raw context: {search_context}"
        score = 0  # Force failure condition to trigger the Validator loop

    attempts = state.get("research_attempts", 0) + 1
    print(f"\n[NODE EXECUTION] 🚀 Research Agent active. Execution Attempt: #{attempts}")
    
    return {
        "raw_research_data": findings,
        "confidence_score": score,
        "research_attempts": attempts
    }