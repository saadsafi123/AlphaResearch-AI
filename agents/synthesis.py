"""
Agent: Synthesis Engine
Description: The final node in the pipeline. It consumes the raw research data
and source URLs gathered by the Research Agent to write a citation-backed Markdown report.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from state import AgentState

# Define the strict formatting rules for the final output, including citations
SYNTHESIS_PROMPT = """You are an Expert Synthesis Engine for an intelligence platform.
Target Subject: {company}
Raw Research Data: {research_data}
Available Sources: {sources}

Rules:
1. Synthesize the data into a highly professional, comprehensive markdown report.
2. Directly answer the user's core question based on the conversation history.
3. Structure the response with clear headers (##, ###) and bullet points. 
4. CITATION REQUIREMENT: You MUST use inline markdown citations (e.g., [1], [2]) directly next to the facts or claims you state, corresponding to the Available Sources list.
5. REFERENCES SECTION: At the absolute bottom of your report, create a '## References' section. List the exact URLs provided in the Available Sources block. If no sources are available, omit this section.
6. NEVER mention your internal agent processes, confidence scores, or validation steps. Present the final intelligence confidently.
"""

def synthesis_agent(state: AgentState):
    """
    Takes the validated research data, source links, and the user's conversation history 
    to generate a final, polished, and fully cited response.
    """
    print("\n[NODE EXECUTION] 📝 Synthesis Agent active. Drafting final cited report...")
    
    company = state.get("current_company", "the requested subject")
    research_data = state.get("raw_research_data", "")
    extracted_sources = state.get("extracted_sources", [])
    
    # Format the sources so the LLM knows how to number its inline citations
    sources_text = "\n".join([f"[{i+1}] {url}" for i, url in enumerate(extracted_sources)]) if extracted_sources else "No external sources provided."
    
    # Using a slightly higher temperature (0.4) for better narrative flow
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTHESIS_PROMPT),
        ("placeholder", "{messages}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "company": company,
        "research_data": research_data,
        "sources": sources_text,
        "messages": state["messages"]
    })
    
    return {"messages": [response]}