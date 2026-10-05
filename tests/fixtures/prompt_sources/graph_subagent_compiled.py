from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

model = init_chat_model("openai:gpt-4o-mini")

RESEARCHER_PROMPT = "You research the order history before anyone answers."


def lookup_order(order_id: str) -> str:
    return f"order {order_id} shipped"


def summarize(state: MessagesState) -> dict:
    system = SystemMessage(content="Summarize the findings in three bullets.")
    return {"messages": [model.invoke([system, *state["messages"]])]}


research_builder = StateGraph(MessagesState)
research_builder.add_node("summarize", summarize)
research_builder.add_edge(START, "summarize")
research_builder.add_edge("summarize", END)
research_graph = research_builder.compile()

investigator = create_react_agent(model, [lookup_order], prompt=RESEARCHER_PROMPT)

builder = StateGraph(MessagesState)
builder.add_node("research", research_graph)
builder.add_node("investigate", investigator)
builder.add_edge(START, "research")
builder.add_edge("research", "investigate")
builder.add_edge("investigate", END)
graph = builder.compile()
