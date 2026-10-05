import os

from langchain_core.prompts import PromptTemplate

BILLING_PROMPT = (
    "Call the billing API with the key "
    + "sk-"
    + "test0fake0key0for0discovery0only00"
    + " and answer {question}."
)

STRIPE_PROMPT = PromptTemplate.from_template(
    "Refund with key " + os.environ.get("STRIPE_KEY", "") + " for {invoice}."
)
