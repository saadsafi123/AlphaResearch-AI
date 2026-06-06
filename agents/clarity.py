"""
Agent: Clarity Gatekeeper (Universal Tier)
Description: Evaluates the user's latest query to determine if the target 
subject is clear. Handles context switching and ambiguous prompt interruptions across any domain.
"""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from state import AgentState

class ClarityDecision(BaseModel):
    """Structured output schema for the Clarity Agent's routing logic."""
    is_clear: str = Field(
        description="Respond 'yes' if you can clearly identify a specific subject, technology, person, event, or concept the user wants to research. Respond 'no' ONLY if the query is completely meaningless or unparseable."
    )
    research_target: str | None = Field(
        description="The exact topic or core target of the research (e.g., 'Transformer Architectures', 'Nvidia', 'History of Silk Road', or 'Photosynthesis')."
    )
    clarification_message: str | None = Field(
        description="Only populate if is_clear is 'no'. Politely ask the user to specify what objective or topic they want to research."
    )

CLARITY_PROMPT = """You are the Universal Gatekeeper for an advanced AI research system. 
Your job is to identify the primary TARGET or SUBJECT of the user's research request in their LATEST message.

Rules for routing:
1. DOMAIN AGNOSTIC: Accept any valid topic (e.g., Science, Technology, Business, History, Engineering, Philosophy, or Current Events). It does NOT have to be a company.
2. FOCUS ON THE LATEST MESSAGE: If the user suddenly shifts topics (e.g., from "Quantum computing" to "Tesla stock layout"), ignore the past topic, mark 'yes', and extract the new target.
3. PRONOUN & HISTORY RESOLUTION: If they ask a follow-up using pronouns (e.g., "how does it work?", "compare them"), analyze the chat history to resolve what "it" or "them" refers to, then mark 'yes'.
4. ABANDONMENT: If the user types an empty phrase or expressions like "stop", "nevermind", or "leave it" without providing an alternative topic, mark 'no' and generate a helpful clarification message asking for a new topic.
5. NO GATEKEEPING BROAD TOPICS: Allow highly comprehensive or broad research tasks to pass. Let the downstream research agents organize the massive data payload.
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
    
    # Map the universal properties to your graph's execution flow
    if decision.is_clear.lower() == "yes":
        return {
            "clarity_status": "clear",
            "current_company": decision.research_target # Keeping key same as state.py for compatibility
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
