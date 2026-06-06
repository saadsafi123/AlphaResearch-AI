"""
Agent: Research Engine (Universal Citation-Backed Tier)
Description: Interfaces with the Tavily search tool to retrieve live, factual data. 
Captures primary source URLs for generation citations and computes content confidence scores.
"""

from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from state import AgentState

class ResearchEvaluation(BaseModel):
    """Structured output schema for the Research Agent."""
    findings: str = Field(description="Comprehensive and deeply structured synthesis of the retrieved data, maintaining technical depth.")
    confidence_score: int = Field(description="Score from 0 to 10 rating the thoroughness and availability of source data.")

RESEARCH_PROMPT = """You are an Elite Analytical Researcher capable of dissecting any complex domain.
Target Subject: {target_topic}

Analyze the live web search context provided below. Extract granular, verified facts, timelines, metrics, or technical nuances.
Structure your findings comprehensively. Rate your confidence (0-10) based on how reliably the retrieved data satisfies the research objective."""

def research_agent(state: AgentState):
    """
    Executes a multi-source web search based on the target entity, extracts raw contexts,
    isolates source URLs for citation handling, and builds analytical findings.
    """
    target = state.get("current_company") # Reads what the gatekeeper passed
    user_query = state["messages"][-1].content if state["messages"] else ""
    
    # Dynamically build a smart, universal query without hardcoded financial keywords
    if target and target.lower() not in user_query.lower():
        search_query = f"{target} {user_query}".strip()
    else:
        search_query = user_query.strip() or f"{target} overview"

    search_tool = TavilySearchResults(max_results=5)
    extracted_sources = []
    
    # Safely invoke the external search API
    try:
        raw_results = search_tool.invoke({"query": search_query})
    except Exception as e:
        print(f"Tavily Tool Error: {e}")
        raw_results = f"Search execution failed: {str(e)}"

    # Process search context while cleanly capturing source URLs for citations
    if isinstance(raw_results, str):
        search_context = f"Raw search context info: {raw_results}"
    elif isinstance(raw_results, list) and len(raw_results) > 0:
        context_blocks = []
        for r in raw_results:
            if isinstance(r, dict):
                url = r.get("url", "")
                content = r.get("content", "")
                if url:
                    extracted_sources.append(url)
                context_blocks.append(f"Source: {url}\nContent: {content}")
            elif isinstance(r, str):
                context_blocks.append(f"Result: {r}")
        search_context = "\n\n".join(context_blocks)
    else:
        search_context = "No accessible research context could be retrieved for this query."

    # Initialize the high-intelligence reasoning model
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1) 
    structured_llm = llm.with_structured_output(ResearchEvaluation)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESEARCH_PROMPT),
        ("user", "Live Web Data Context:\n{context}\nGenerate objective findings and a confidence score.")
    ])
    
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({"target_topic": target or "Requested Subject", "context": search_context})
        findings = result.findings
        score = result.confidence_score
    except Exception as e:
        print(f"LLM Parsing Error: {e}")
        findings = f"Architectural parsing conflict. Raw verified context: {search_context}"
        score = 0 
        
    attempts = state.get("research_attempts", 0) + 1
    print(f"\n[NODE EXECUTION] 🚀 Research Agent active. Target: '{target}'. Execution Attempt: #{attempts}")
    
    # Return the findings along with the pristine list of sources to the global state
    return {
        "raw_research_data": findings,
        "confidence_score": score,
        "research_attempts": attempts,
        "extracted_sources": list(set(extracted_sources)) # Deduplicate URLs seamlessly
    }