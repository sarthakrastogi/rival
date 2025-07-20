from typing import Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from IPython.display import display, Image
from langchain_core.runnables.graph import MermaidDrawMethod
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class TravelState(TypedDict):
    user_query: str
    intent: str
    mood: str
    suggestion: str


def detect_intent(state: TravelState) -> TravelState:
    """Detect the user's travel intent."""
    prompt = ChatPromptTemplate.from_template(
        "The user said: {user_query}\n"
        "Classify the intent into one of the following: 'Destination Recommendation', 'Travel Tips', 'Visa Information'."
    )
    chain = prompt | ChatOpenAI(temperature=0)
    intent = chain.invoke({"user_query": state["user_query"]}).content.strip()
    return {"intent": intent}


def detect_mood(state: TravelState) -> TravelState:
    """Detect user's mood based on their travel preferences."""
    prompt = ChatPromptTemplate.from_template(
        "Based on this query: {user_query}\n"
        "Classify the user's travel mood as 'Adventurous', 'Relaxed', or 'Curious'."
    )
    chain = prompt | ChatOpenAI(temperature=0)
    mood = chain.invoke({"user_query": state["user_query"]}).content.strip()
    return {"mood": mood}


def suggest_destination(state: TravelState) -> TravelState:
    """Suggest travel destinations."""
    prompt = ChatPromptTemplate.from_template(
        "User mood: {mood}. Provide a destination recommendation for the query: {user_query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    suggestion = chain.invoke(
        {"user_query": state["user_query"], "mood": state["mood"]}
    ).content.strip()
    return {"suggestion": suggestion}


def give_travel_tips(state: TravelState) -> TravelState:
    """Give travel tips."""
    prompt = ChatPromptTemplate.from_template(
        "User mood: {mood}. Provide travel tips based on the query: {user_query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    suggestion = chain.invoke(
        {"user_query": state["user_query"], "mood": state["mood"]}
    ).content.strip()
    return {"suggestion": suggestion}


def provide_visa_info(state: TravelState) -> TravelState:
    """Provide visa information."""
    prompt = ChatPromptTemplate.from_template(
        "Provide visa-related information based on the query: {user_query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    suggestion = chain.invoke({"user_query": state["user_query"]}).content.strip()
    return {"suggestion": suggestion}


def ask_for_clarification(state: TravelState) -> TravelState:
    """Escalate or clarify unclear queries."""
    return {
        "suggestion": "Your query is a bit unclear. Could you please rephrase or provide more details?"
    }


def route_by_intent(state: TravelState) -> str:
    """Decide next step based on detected intent."""
    if state["intent"] == "Destination Recommendation":
        return "suggest_destination"
    elif state["intent"] == "Travel Tips":
        return "give_travel_tips"
    elif state["intent"] == "Visa Information":
        return "provide_visa_info"
    else:
        return "ask_for_clarification"


# Define the graph
travel_graph = StateGraph(TravelState)

# Nodes
travel_graph.add_node("detect_intent", detect_intent)
travel_graph.add_node("detect_mood", detect_mood)
travel_graph.add_node("suggest_destination", suggest_destination)
travel_graph.add_node("give_travel_tips", give_travel_tips)
travel_graph.add_node("provide_visa_info", provide_visa_info)
travel_graph.add_node("ask_for_clarification", ask_for_clarification)

# Edges
travel_graph.add_edge("detect_intent", "detect_mood")
travel_graph.add_conditional_edges(
    "detect_mood",
    route_by_intent,
    {
        "suggest_destination": "suggest_destination",
        "give_travel_tips": "give_travel_tips",
        "provide_visa_info": "provide_visa_info",
        "ask_for_clarification": "ask_for_clarification",
    },
)

# End nodes
travel_graph.add_edge("suggest_destination", END)
travel_graph.add_edge("give_travel_tips", END)
travel_graph.add_edge("provide_visa_info", END)
travel_graph.add_edge("ask_for_clarification", END)

# Set entry point
travel_graph.set_entry_point("detect_intent")

# Compile
travel_agent = travel_graph.compile()

# Draw the graph
# display(
#    Image(travel_agent.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API))
# )


def handle_travel_query(user_query: str) -> Dict[str, str]:
    """Run the travel assistant agent."""
    result = travel_agent.invoke({"user_query": user_query})
    return {
        "intent": result.get("intent", ""),
        "mood": result.get("mood", ""),
        "suggestion": result.get("suggestion", ""),
    }
