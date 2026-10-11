from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent


def lookup_order(order_id: str) -> str:
    return f"order {order_id}: shipped"


researcher = create_react_agent(
    "openai:gpt-4o-mini",
    [lookup_order],
    prompt='You research orders. Reply with JSON like {"status": "..."}.',
)

refunds = create_agent(
    "openai:gpt-4o-mini",
    tools=[lookup_order],
    system_prompt=SystemMessage(content="You handle refunds within policy."),
)


def _tiered_prompt(state: dict) -> list:
    return [SystemMessage(content=f"Customer tier: {state['tier']}"), *state["messages"]]


concierge = create_react_agent("openai:gpt-4o-mini", [lookup_order], prompt=_tiered_prompt)
