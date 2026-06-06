"""
Agent: Clarity Gatekeeper
Description: Evaluates the user's latest query to determine if the target 
subject is clear. Handles context switching and ambiguous prompt interruptions.
"""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from state import AgentState

class ClarityDecision(BaseModel):
    """Structured output schema for the Clarity Agent's routing logic."""
    is_clear: str = Field(
        description="Respond 'yes' if you can determine the company, person, or general topic the user wants to research. Respond 'no' ONLY if the query is completely meaningless."
    )
    company_name: str | None = Field(
        description="The target of the research (e.g., 'Tesla', 'Microsoft and Salesforce', or a general topic like 'AI Agents')."
    )
    clarification_message: str | None = Field(
        description="Only populate if is_clear is 'no'. Ask what new topic they want to research."
    )

CLARITY_PROMPT = """You are the Gatekeeper for an AI research system. 
Your job is to identify the TARGET of the user's research in their LATEST message.

Rules for routing:
1. FOCUS ON THE LATEST MESSAGE: If the user pivots to a new topic (e.g., "tell me about AI agents", "Analyze Nvidia"), ignore past confusion. Mark 'yes' and extract the new topic.
2. ACCEPT GENERAL TOPICS: If the user asks about a general technology or concept (e.g., "AI agents"), treat that as the target and mark 'yes'. It does not have to be a specific company.
3. FOLLOW-UPS: If they use pronouns ("compare them"), look at history to resolve the names. Mark 'yes'.
4. ABANDONMENT: If the user says "leave it", "stop", or "nevermind" WITHOUT providing a new topic, mark 'no' and gently ask: "No problem! What new topic or company would you like to research today?"
5. DO NOT BLOCK broad questions. Let the research agent do the heavy lifting.
"""

def clarity_agent(state: AgentState):
    """
    Analyzes the conversation state and routes to either the Research Agent 
    or pauses execution to ask the user for clarification.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(ClarityDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", CLARITY_PROMPT),
        ("placeholder", "{messages}")
    ])
    
    chain = prompt | structured_llm
    decision = chain.invoke({"messages": state["messages"]})
    
    # Map the string response to the system's execution flags
    if decision.is_clear.lower() == "yes":
        return {
            "clarity_status": "clear",
            "current_company": decision.company_name
        }
    else:
        return {
            "clarity_status": "needs_clarification",
            "messages": [AIMessage(content=decision.clarification_message)] 
        }
    

# class ClarityDecision(BaseModel):
#     # Changed from bool to str to prevent JSON parsing errors
#     is_clear: str = Field(
#         description="Respond 'yes' if you can determine which company or companies the user wants to talk about (either explicitly or from history). Respond 'no' ONLY if the query is completely blind or missing any company context."
#     )
#     company_name: str | None = Field(
#         description="The name of the company or companies identified (e.g., 'Tesla', 'Stripe', 'Tesla and Stripe')."
#     )
#     clarification_message: str | None = Field(
#         description="Only populate this if is_clear is 'no'. Ask a friendly question to find out what business or company they want to discuss."
#     )

# CLARITY_PROMPT = """You are the Clarity Gatekeeper for a business research network. 
# Your sole responsibility is to make sure we know WHICH companies are being discussed.

# Rules for Clarity evaluation:
# 1. If the user mentions a business, company, or famous business leader (like Elon Musk/Tesla, Stripe, Apple), mark 'yes'.
# 2. If the user asks a follow-up, comparison, or broad question (e.g., "compare them", "tell me about their revenue", "what about the other one?"), look closely at the chat history. If the names of the companies were established earlier, mark 'yes' and pass those company names along.
# 3. Do NOT block a request just because it is broad (like "compare both" or "tell me about their revenue"). Let the research agent handle the depth.
# 4. Mark 'no' ONLY if the user's message contains absolutely no business entity context and the chat history doesn't clear it up (e.g., "Tell me something interesting", "Do you know a person?").
# """
