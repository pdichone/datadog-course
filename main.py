"""
First Traced LLM Call
Demonstrates basic auto-instrumentation with OpenAI
"""

import os
from openai import OpenAI
from ddtrace.llmobs import LLMObs

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


def main():
    # Initialize LLM Observability
    LLMObs.enable(
        ml_app=os.getenv("DD_LLMOBS_ML_APP", "my-llm-app"),
        agentless_enabled=True,
        api_key=os.getenv("DD_API_KEY"),
        site=os.getenv("DD_SITE", "datadoghq.com"),
    )

    # Create OpenAI client
    client = OpenAI()

    # Make a traced LLM call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain Newton's first law in simple terms."},
        ],
        max_tokens=150,
    )

    print("Response:", response.choices[0].message.content)
    print(f"Tokens used: {response.usage.total_tokens}")

    # Shutdown gracefully
    LLMObs.disable()


if __name__ == "__main__":
    main()
