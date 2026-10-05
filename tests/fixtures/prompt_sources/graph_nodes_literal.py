from app.prompts import WRITER_PROMPT
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph

model = init_chat_model("openai:gpt-4o-mini")

PLANNER_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", "Plan the next support step for {goal}."),
        ("human", "{question}"),
    ]
)


def planner(state: MessagesState) -> dict:
    question = state["messages"][-1].content
    messages = PLANNER_TEMPLATE.format_messages(goal="support", question=question)
    return {"messages": [model.invoke(messages)]}


def reviewer(state: MessagesState) -> dict:
    system = SystemMessage(content="Check the draft for tone and accuracy.")
    return {"messages": [model.invoke([system, *state["messages"]])]}


def writer(state: MessagesState) -> dict:
    question = state["messages"][0].content
    return {"messages": [model.invoke(WRITER_PROMPT.format(question=question))]}


builder = StateGraph(MessagesState)
builder.add_node("planner", planner)
builder.add_node("reviewer", reviewer)
builder.add_node("writer", writer)
builder.add_edge(START, "planner")
builder.add_edge("planner", "reviewer")
builder.add_edge("reviewer", "writer")
builder.add_edge("writer", END)
graph = builder.compile()
