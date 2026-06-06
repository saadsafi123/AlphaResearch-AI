"""
Agent: Synthesis Engine
Description: The final node in the pipeline. It consumes the raw research data
gathered by the Research Agent and formats it into a professional, easily 
readable Markdown report tailored to the user's original query.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from state import AgentState

# Define the strict formatting rules for the final output
SYNTHESIS_PROMPT = """You are an expert executive assistant.
Target Subject: {company}
Raw Research Data: {research_data}

Rules:
1. Synthesize the data into a professional markdown report.
2. Directly answer the user's core question based on the conversation history.
3. Structure the response with clear headers (##, ###) and bullet points. 
4. NEVER mention your internal agent processes, confidence scores, or validation steps. Present the final facts confidently.
"""

def synthesis_agent(state: AgentState):
    """
    Takes the validated research data and the user's conversation history 
    to generate a final, polished response.
    """
    print("\n[NODE EXECUTION] 📝 Synthesis Agent active. Drafting final report...")
    
    company = state.get("current_company", "the requested subject")
    research_data = state.get("raw_research_data", "")
    
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
        "messages": state["messages"]
    })
    
    return {"messages": [response]}