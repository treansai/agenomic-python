from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph

model = init_chat_model("openai:gpt-4o-mini")

ESCALATION_PROMPT = "Escalate to a human when the customer asks twice."


def escalate(state: MessagesState) -> dict:
    reply = model.invoke([SystemMessage(content=ESCALATION_PROMPT), *state["messages"]])
    return {"messages": [reply]}


class Nodes:
    def apologize(self, state: MessagesState) -> dict:
        system = SystemMessage(content="Apologize once and offer a fix.")
        return {"messages": [model.invoke([system, *state["messages"]])]}


def build(prefix: str) -> object:
    nodes = Nodes()
    builder = StateGraph(MessagesState)
    builder.add_node(f"{prefix}_escalate", escalate)
    builder.add_node("apologize", nodes.apologize)
    builder.add_edge(START, f"{prefix}_escalate")
    builder.add_edge(f"{prefix}_escalate", "apologize")
    builder.add_edge("apologize", END)
    return builder.compile()
