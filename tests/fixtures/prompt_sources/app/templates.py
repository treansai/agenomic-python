from datetime import datetime, timezone

from langchain import hub
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

from app.prompts import PLANNER_SYSTEM_PROMPT, SIGNATURE_TEMPLATE


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", PLANNER_SYSTEM_PROMPT),
        MessagesPlaceholder("history", optional=True),
        ("human", "{question}"),
    ]
).partial(locale="en")

SUMMARY_PROMPT = PromptTemplate.from_template(
    "Summarize the ticket in {style} style:\n{ticket}",
    partial_variables={"style": "neutral"},
)

HANDOFF_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="Hand-offs follow the {severity} matrix verbatim."),
        ("placeholder", "{history}"),
        ("user", "{ticket}"),
    ]
)

TRIAGE_PROMPT = PromptTemplate.from_template(
    "Classify {{ticket}} for {{team}}.",
    template_format="mustache",
)

DATED_PROMPT = PromptTemplate(
    template="Today is {today}. Answer {question}.",
    input_variables=["question"],
    partial_variables={"today": _today},
)

REVIEW_PROMPT = hub.pull("support/review-reply")


def build_research_prompt(order_id: str) -> PromptTemplate:
    return PromptTemplate.from_template(f"Research order {order_id} and list {{facts}}.")


def signature(agent_name: str) -> SystemMessage:
    return SystemMessage(content=SIGNATURE_TEMPLATE.format(agent_name=agent_name))
