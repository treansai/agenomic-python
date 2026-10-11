import textwrap

PLANNER_SYSTEM_PROMPT = (
    "You plan support work for {customer_name}.\nAnswer in {locale} and never promise refunds."
)

WRITER_PROMPT = textwrap.dedent(
    """\
    Write a short answer to {question}.
    Keep it under 120 words.
    """
)

ROUTER_PROMPT = "Route the request to billing, shipping or returns."

SIGNATURE_TEMPLATE = "Signed by {agent_name}, support team."

COMPANY = "Acme"

GREETING_TEMPLATE = "Welcome to {company} support."

BANNER_PROMPT = GREETING_TEMPLATE.format(company=COMPANY)

MAX_TURNS = 8
