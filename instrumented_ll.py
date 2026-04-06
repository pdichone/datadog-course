"""
Instrumented LLM Calls
Demonstrates custom span creation and annotation
"""

from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import llm
from openai import OpenAI
from typing import Optional
import os

from dotenv import load_dotenv

load_dotenv()

app_name = os.getenv("DD_LLMOBS_ML_APP", "my-llm-app")


LLMObs.enable(
    ml_app=app_name,
    agentless_enabled=True,
    api_key=os.getenv("DD_API_KEY"),
    site=os.getenv("DD_SITE", "datadoghq.com"),
)
client = OpenAI()


@llm(model_name="gpt-4o-mini", model_provider="openai")
def generate_summary(document: str, max_length: int = 200) -> str:
    """
    Generate a concise summary of the given document.

    Args:
        document: The text to summarize
        max_length: Maximum tokens in response

    Returns:
        Summary string
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a professional summarizer. Create concise, accurate summaries.",
            },
            {"role": "user", "content": f"Summarize this document:\n\n{document}"},
        ],
        max_tokens=max_length,
    )

    summary = response.choices[0].message.content

    # Annotate with additional metadata
    LLMObs.annotate(
        metadata={
            "document_length": len(document),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / len(document) if document else 0,
        }
    )

    return summary


@llm(model_name="gpt-4o-mini", model_provider="openai")
def classify_intent(
    user_message: str, user_id: Optional[str] = None, session_id: Optional[str] = None
) -> dict:
    """
    Classify user intent for routing.

    Args:
        user_message: The user's message
        user_id: Optional user identifier
        session_id: Optional session identifier

    Returns:
        Dict with intent and confidence
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Classify the user's intent into one of these categories:
                - billing: Questions about invoices, payments, pricing
                - technical: Technical issues, bugs, how-to questions
                - account: Account management, settings, profile
                - sales: Interest in purchasing, upgrades, enterprise
                - other: Anything else

                Respond with JSON: {"intent": "category", "confidence": 0.0-1.0}""",
            },
            {"role": "user", "content": user_message},
        ],
        max_tokens=50,
        response_format={"type": "json_object"},
    )

    import json

    result = json.loads(response.choices[0].message.content)

    # Add rich metadata for filtering
    LLMObs.annotate(
        input_data=user_message,
        output_data=result,
        metadata={
            "user_id": user_id,
            "session_id": session_id,
            "detected_intent": result.get("intent"),
            "confidence": result.get("confidence"),
        },
        tags={
            "intent_type": result.get("intent", "unknown"),
            "high_confidence": str(result.get("confidence", 0) > 0.8),
        },
    )

    return result


if __name__ == "__main__":
    # Test summary generation
    doc = """
    Artificial intelligence has transformed how businesses operate.
    Machine learning models now power everything from customer service
    to fraud detection. However, the rapid adoption of AI has created
    new challenges around monitoring, debugging, and cost management.
    """

    summary = generate_summary(doc)
    print(f"Summary: {summary}")

    # Test intent classification
    messages = [
        "I can't log into my account",
        "How much does the enterprise plan cost?",
        "The API is returning 500 errors",
    ]

    for msg in messages:
        result = classify_intent(msg, user_id="user_123", session_id="sess_456")
        print(f"'{msg}' -> {result}")

    LLMObs.disable()
