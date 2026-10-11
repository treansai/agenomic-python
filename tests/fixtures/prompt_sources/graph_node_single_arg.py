from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph

model = init_chat_model("openai:gpt-4o-mini")

TRIAGE_SYSTEM_PROMPT = "Sort the request into billing, shipping or returns."


def triage(state: MessagesState) -> dict:
    reply = model.invoke([SystemMessage(content=TRIAGE_SYSTEM_PROMPT), *state["messages"]])
    return {"messages": [reply]}


workflow = StateGraph(MessagesState)
workflow.add_node(triage)
workflow.add_edge(START, "triage")
workflow.add_edge("triage", END)
app = workflow.compile()
