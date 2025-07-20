from typing import Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class SupportFlowState(TypedDict):
    customer_input: str
    intent: str
    reply: str


# Shared policy document for the finance chatbot
FINANCE_CHATBOT_KNOWLEDGE = """
You are a finance support assistant for FinSavvy, a modern fintech company.
- If the intent is Complaint: Acknowledge the issue professionally, explain next steps based on company policy, and reassure the customer.
- If the intent is Feedback: Thank the customer for their feedback and note how it may be used to improve services.
- If the intent is Inquiry: Provide a clear, concise answer using FinSavvy's FAQ and product information.

Example FAQs:
- Account opening takes 1-2 business days.
- There are no hidden fees; our fee structure is transparent.
- For loan applications, approval typically takes 3-5 business days.
Always respond politely and avoid making commitments outside company policy.
"""


# Intent detection
def detect_intent(state: SupportFlowState) -> SupportFlowState:
    """Detects intent: Complaint, Feedback, or Inquiry."""
    prompt = ChatPromptTemplate.from_template(
        "Identify the customer's intent: Complaint, Feedback, or Inquiry.\nMessage: {customer_input}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    intent = chain.invoke({"customer_input": state["customer_input"]}).content.strip()
    return {"intent": intent}


# Complaint handler
def handle_complaint(state: SupportFlowState) -> SupportFlowState:
    """Handles customer complaints with FinSavvy's policy."""
    prompt = ChatPromptTemplate.from_template(
        "{knowledge}\n\nCustomer Complaint: {customer_input}\n\nDraft a professional response."
    )
    chain = prompt | ChatOpenAI(temperature=0)
    reply = chain.invoke(
        {
            "knowledge": FINANCE_CHATBOT_KNOWLEDGE,
            "customer_input": state["customer_input"],
        }
    ).content.strip()
    return {"reply": reply}


# Feedback handler
def handle_feedback(state: SupportFlowState) -> SupportFlowState:
    """Handles customer feedback politely."""
    prompt = ChatPromptTemplate.from_template(
        "{knowledge}\n\nCustomer Feedback: {customer_input}\n\nDraft a polite thank-you response."
    )
    chain = prompt | ChatOpenAI(temperature=0)
    reply = chain.invoke(
        {
            "knowledge": FINANCE_CHATBOT_KNOWLEDGE,
            "customer_input": state["customer_input"],
        }
    ).content.strip()
    return {"reply": reply}


# Inquiry handler
def handle_inquiry(state: SupportFlowState) -> SupportFlowState:
    """Answers customer inquiries using FAQs."""
    prompt = ChatPromptTemplate.from_template(
        "{knowledge}\n\nCustomer Inquiry: {customer_input}\n\nProvide a clear, concise answer."
    )
    chain = prompt | ChatOpenAI(temperature=0)
    reply = chain.invoke(
        {
            "knowledge": FINANCE_CHATBOT_KNOWLEDGE,
            "customer_input": state["customer_input"],
        }
    ).content.strip()
    return {"reply": reply}


# Intent router — no escalation, always answer
def intent_router(state: SupportFlowState) -> str:
    """Routes intent to the correct response node."""
    if state["intent"] == "Complaint":
        return "handle_complaint"
    elif state["intent"] == "Feedback":
        return "handle_feedback"
    else:
        return "handle_inquiry"


# Graph definition
graph = StateGraph(SupportFlowState)

# Nodes
graph.add_node("detect_intent", detect_intent)
graph.add_node("handle_complaint", handle_complaint)
graph.add_node("handle_feedback", handle_feedback)
graph.add_node("handle_inquiry", handle_inquiry)

# Edges
graph.add_conditional_edges(
    "detect_intent",
    intent_router,
    {
        "handle_complaint": "handle_complaint",
        "handle_feedback": "handle_feedback",
        "handle_inquiry": "handle_inquiry",
    },
)
graph.add_edge("handle_complaint", END)
graph.add_edge("handle_feedback", END)
graph.add_edge("handle_inquiry", END)

graph.set_entry_point("detect_intent")

# Compile graph
support_app = graph.compile()


def process_customer_message(inputs: Dict) -> Dict[str, str]:
    """Runs the FinSavvy support assistant flow."""
    result = support_app.invoke(inputs)
    return {
        "intent": result["intent"],
        "reply": result["reply"],
    }


if __name__ == "__main__":
    # Example usage
    test_input = {
        "customer_input": "I'm really unhappy about the extra charge on my account."
    }
    output = process_customer_message(test_input)
    print("Intent:", output["intent"])
    print("Reply:", output["reply"])
