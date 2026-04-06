import os
import json
from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import llm
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()

app_name = os.getenv("DD_LLMOBS_ML_APP", "my-llm-app")
client = OpenAI()


LLMObs.enable(
    ml_app=app_name,
    agentless_enabled=True,
    api_key=os.getenv("DD_API_KEY"),
    app_key=os.getenv("DD_APP_KEY"),
    site=os.getenv("DD_SITE", "datadoghq.com"),
)


def scrub_pii(text: str) -> str:
    """Remove PII before tracing."""
    # SSN
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]", text)

    # Email
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL REDACTED]", text
    )

    # Credit card (basic pattern)
    text = re.sub(
        r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CARD REDACTED]", text
    )

    return text


def generate_response(message: str) -> str:
    """Generate a response using OpenAI."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content


# @llm(model_name="gpt-4o-mini", model_provider="openai")
def process_user_message(message: str) -> str:
    response = generate_response(message)

    # Scrub before annotation
    # LLMObs.annotate(input_data=scrub_pii(message), output_data=scrub_pii(response))

    return response


if __name__ == "__main__":
    test_queries = [
        "Hi, I need help with my travel documents. "
        "My US passport number is 123456789, "
        "my California driver's license is Y1234567, "
        "and you can reach me at john.smith@gmail.com. "
        "Can you verify my identity?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(generate_response(query))
        # print(f"--- Query {i} ---")
        # print(f"ORIGINAL:  {query}")
        # print(f"SCRUBBED:  {process_user_message(query)}")
        # print()

    LLMObs.disable()
