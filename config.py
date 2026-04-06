"""
Datadog LLM Observability Configuration

This module handles initialization of Datadog LLM Observability
with support for both agentless and agent-based deployments.
"""

import os
from ddtrace.llmobs import LLMObs


def init_llm_observability(app_name: str = "dd-tester", agentless: bool = True) -> None:
    """
    Initialize Datadog LLM Observability.

    Args:
        app_name: Name to identify this app in Datadog
        agentless: Use agentless mode (True for dev, False for prod)
    """
    LLMObs.enable(
        ml_app=app_name,
        api_key=os.getenv("DD_API_KEY"),
        site=os.getenv("DD_SITE", "datadoghq.com"),
        agentless_enabled=agentless,
    )
    print(f"LLM Observability enabled for {app_name}")


def shutdown_llm_observability() -> None:
    """Gracefully shutdown LLM Observability."""
    LLMObs.disable()
