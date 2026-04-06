"""
Customer Support Agent with Full Instrumentation
Demonstrates agent tracing with tools and decision routing
"""

from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import agent, tool, llm, task
from openai import OpenAI
from typing import Dict, Any, Optional
import json
import os

from dotenv import load_dotenv

load_dotenv()


app_name = os.getenv("DD_LLMOBS_ML_APP", "my-llm-app")
client = OpenAI()


LLMObs.enable(
    ml_app=app_name,
    agentless_enabled=True,
    api_key=os.getenv("DD_API_KEY"),
    site=os.getenv("DD_SITE", "datadoghq.com"),
)

# Simulated database
ORDERS_DB = {
    "user_001": {
        "order_id": "ORD-12345",
        "status": "delivered",
        "items": ["Widget Pro", "Widget Lite"],
        "total": 149.99,
        "date": "2026-02-15",
        "refundable": True,
    },
    "user_002": {
        "order_id": "ORD-67890",
        "status": "shipped",
        "items": ["Enterprise Suite"],
        "total": 999.99,
        "date": "2026-03-01",
        "refundable": True,
    },
}


@agent
def customer_support_agent(
    user_query: str, user_id: str, session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    AI-powered customer support agent.

    Handles:
    - Order status inquiries
    - Refund requests
    - General questions

    Args:
        user_query: The customer's question
        user_id: Customer identifier
        session_id: Optional conversation session ID

    Returns:
        Dict with response and metadata
    """
    # Annotate agent with context
    LLMObs.annotate(
        input_data=user_query,
        metadata={"user_id": user_id, "session_id": session_id, "agent_version": "2.0"},
    )

    # Step 1: Classify intent
    intent_result = classify_intent(user_query)
    intent = intent_result["intent"]
    confidence = intent_result["confidence"]

    # Step 2: Route based on intent
    if intent == "order_status" and confidence > 0.7:
        order_info = lookup_order(user_id)
        if order_info:
            response = generate_order_response(user_query, order_info)
        else:
            response = "I couldn't find any orders for your account. Can you verify your account details?"

    elif intent == "refund_request" and confidence > 0.7:
        order_info = lookup_order(user_id)
        if order_info and order_info.get("refundable"):
            refund_result = process_refund(order_info["order_id"], user_id)
            response = generate_refund_response(refund_result)
        elif order_info:
            response = "Unfortunately, this order is not eligible for a refund."
        else:
            response = (
                "I couldn't find an order to refund. Can you provide your order ID?"
            )

    elif intent == "billing":
        response = generate_billing_response(user_query)

    else:
        # General inquiry
        response = generate_general_response(user_query)

    result = {
        "response": response,
        "intent": intent,
        "confidence": confidence,
        "user_id": user_id,
    }

    LLMObs.annotate(
        output_data=result,
        metadata={
            "intent_handled": intent,
            "confidence": confidence,
            "response_length": len(response),
        },
    )

    return result


@llm(model_name="gpt-4o-mini", model_provider="openai")
def classify_intent(query: str) -> Dict[str, Any]:
    """Classify user intent for routing."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Classify the customer inquiry into one of these intents:
                - order_status: Questions about order tracking, delivery, status
                - refund_request: Requests to return items or get refunds
                - billing: Invoice, payment, or pricing questions
                - technical: Product issues or how-to questions
                - general: Other inquiries

                Respond with JSON: {"intent": "category", "confidence": 0.0-1.0}""",
            },
            {"role": "user", "content": query},
        ],
        max_tokens=50,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)

    LLMObs.annotate(
        input_data=query,
        output_data=result,
        metadata={
            "classified_intent": result["intent"],
            "confidence": result["confidence"],
        },
    )

    return result


@tool
def lookup_order(user_id: str) -> Optional[Dict[str, Any]]:
    """Look up user's most recent order."""
    order = ORDERS_DB.get(user_id)

    LLMObs.annotate(
        metadata={
            "user_id": user_id,
            "order_found": order is not None,
            "order_id": order["order_id"] if order else None,
        }
    )

    return order


@tool
def process_refund(order_id: str, user_id: str) -> Dict[str, Any]:
    """Process refund request."""
    # Simulated refund processing
    result = {
        "status": "approved",
        "refund_id": f"REF-{order_id[-5:]}",
        "amount": ORDERS_DB.get(user_id, {}).get("total", 0),
        "estimated_days": 5,
    }

    LLMObs.annotate(
        metadata={
            "order_id": order_id,
            "refund_status": result["status"],
            "refund_amount": result["amount"],
        }
    )

    return result


@llm(model_name="gpt-4o-mini", model_provider="openai")
def generate_order_response(query: str, order_info: Dict) -> str:
    """Generate response about order status."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""You are a friendly customer support agent.
The customer's order information:
- Order ID: {order_info['order_id']}
- Status: {order_info['status']}
- Items: {', '.join(order_info['items'])}
- Total: ${order_info['total']}
- Date: {order_info['date']}

Provide a helpful response about their order.""",
            },
            {"role": "user", "content": query},
        ],
        max_tokens=200,
    )

    return response.choices[0].message.content


@llm(model_name="gpt-4o-mini", model_provider="openai")
def generate_refund_response(refund_result: Dict) -> str:
    """Generate response about refund status."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""You are a friendly customer support agent.
A refund has been processed:
- Status: {refund_result['status']}
- Refund ID: {refund_result['refund_id']}
- Amount: ${refund_result['amount']}
- Estimated time: {refund_result['estimated_days']} business days

Confirm the refund details to the customer.""",
            },
            {"role": "user", "content": "Please confirm my refund."},
        ],
        max_tokens=150,
    )

    return response.choices[0].message.content


@llm(model_name="gpt-4o-mini", model_provider="openai")
def generate_billing_response(query: str) -> str:
    """Generate response for billing inquiries."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a customer support agent helping with billing.
Our pricing:
- Basic: $29/month
- Pro: $99/month
- Enterprise: Contact sales

Provide helpful billing information.""",
            },
            {"role": "user", "content": query},
        ],
        max_tokens=150,
    )

    return response.choices[0].message.content


@llm(model_name="gpt-4o-mini", model_provider="openai")
def generate_general_response(query: str) -> str:
    """Generate response for general inquiries."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful customer support agent. Be concise and friendly.",
            },
            {"role": "user", "content": query},
        ],
        max_tokens=150,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Test scenarios
    test_cases = [
        ("Where is my order?", "user_001"),
        ("I want a refund for my purchase", "user_001"),
        ("How much does the Pro plan cost?", "user_002"),
        ("What are your business hours?", "user_001"),
    ]

    for query, user_id in test_cases:
        print(f"\n{'='*50}")
        print(f"User: {user_id}")
        print(f"Query: {query}")
        result = customer_support_agent(query, user_id, session_id="test_session")
        print(f"Intent: {result['intent']} ({result['confidence']:.2f})")
        print(f"Response: {result['response']}")

    LLMObs.disable()
