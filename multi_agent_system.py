"""
Multi-Agent System with Orchestration
Demonstrates tracing across multiple coordinated agents
"""

from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import agent, tool, llm, task
from openai import OpenAI
from typing import List, Dict, Any
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


@agent(name="orchestrator")
def orchestrator_agent(task: str) -> Dict[str, Any]:
    """
    Central orchestrator that plans and delegates to worker agents.

    Args:
        task: The user's request

    Returns:
        Final synthesized result
    """
    LLMObs.annotate(
        input_data=task, metadata={"agent_role": "orchestrator", "agent_version": "1.0"}
    )

    # Step 1: Create execution plan
    plan = create_execution_plan(task)

    # Step 2: Execute plan with appropriate agents
    execution_results = []

    for step in plan["steps"]:
        step_type = step["type"]
        step_input = step["input"]

        if step_type == "research":
            result = research_agent(step_input)
        elif step_type == "analyze":
            result = analysis_agent(step_input, execution_results)
        elif step_type == "generate":
            result = generation_agent(step_input, execution_results)
        elif step_type == "validate":
            result = validation_agent(
                execution_results[-1] if execution_results else {}
            )
        else:
            result = {"error": f"Unknown step type: {step_type}"}

        execution_results.append({"step": step_type, "result": result})

    # Step 3: Synthesize final response
    final_result = synthesize_results(task, execution_results)

    LLMObs.annotate(
        output_data=final_result,
        metadata={
            "steps_executed": len(execution_results),
            "agents_used": [r["step"] for r in execution_results],
        },
    )

    return final_result


@llm(model_name="gpt-4o-mini", model_provider="openai")
def create_execution_plan(task: str) -> Dict[str, Any]:
    """Create a plan for executing the task."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a task planner. Break down the task into steps.
Available step types: research, analyze, generate, validate

Respond with JSON:
{
    "steps": [
        {"type": "step_type", "input": "description of what to do"}
    ]
}

Keep it to 2-4 steps maximum.""",
            },
            {"role": "user", "content": task},
        ],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


@agent(name="research")
def research_agent(query: str) -> Dict[str, Any]:
    """Research agent for gathering information."""
    LLMObs.annotate(input_data=query, metadata={"agent_role": "research"})

    # Perform research
    findings = perform_research(query)

    LLMObs.annotate(
        output_data=findings,
        metadata={"findings_count": len(findings.get("facts", []))},
    )

    return findings


@tool
def perform_research(query: str) -> Dict[str, Any]:
    """Simulate research by generating relevant facts."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": 'Generate 3-5 relevant facts about the topic. Respond with JSON: {"facts": [...]}',
            },
            {"role": "user", "content": query},
        ],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


@agent(name="analysis")
def analysis_agent(task: str, context: List[Dict]) -> Dict[str, Any]:
    """Analysis agent for processing and analyzing information."""
    LLMObs.annotate(
        input_data={"task": task, "context_items": len(context)},
        metadata={"agent_role": "analysis"},
    )

    # Analyze the context
    analysis = perform_analysis(task, context)

    return analysis


@llm(model_name="gpt-4o-mini", model_provider="openai")
def perform_analysis(task: str, context: List[Dict]) -> Dict[str, Any]:
    """Perform analysis on gathered information."""
    context_str = json.dumps(context, indent=2)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"Analyze this context and identify key insights.\nContext: {context_str}",
            },
            {"role": "user", "content": task},
        ],
    )

    return {
        "analysis": response.choices[0].message.content,
        "context_analyzed": len(context),
    }


@agent(name="generation")
def generation_agent(task: str, context: List[Dict]) -> Dict[str, Any]:
    """Generation agent for creating content."""
    LLMObs.annotate(
        input_data={"task": task, "context_items": len(context)},
        metadata={"agent_role": "generation"},
    )

    content = generate_content(task, context)

    return {"content": content}


@llm(model_name="gpt-4o-mini", model_provider="openai")
def generate_content(task: str, context: List[Dict]) -> str:
    """Generate content based on task and context."""
    context_str = json.dumps(context, indent=2)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"Based on this context, generate the requested content.\nContext: {context_str}",
            },
            {"role": "user", "content": task},
        ],
    )

    return response.choices[0].message.content


@agent(name="validation")
def validation_agent(content: Dict) -> Dict[str, Any]:
    """Validation agent for reviewing generated content."""
    LLMObs.annotate(input_data=content, metadata={"agent_role": "validation"})

    validation = validate_content(content)

    return validation


@llm(model_name="gpt-4o-mini", model_provider="openai")
def validate_content(content: Dict) -> Dict[str, Any]:
    """Validate generated content for quality and accuracy."""
    content_str = json.dumps(content, indent=2)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Review this content for quality and accuracy.
Respond with JSON:
{
    "approved": true/false,
    "score": 0-10,
    "feedback": "..."
}""",
            },
            {"role": "user", "content": f"Review: {content_str}"},
        ],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


@task(name="synthesize_results")
def synthesize_results(task: str, results: List[Dict]) -> Dict[str, Any]:
    """Synthesize all agent results into final output."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Synthesize these agent results into a coherent final response.",
            },
            {
                "role": "user",
                "content": f"Task: {task}\n\nResults: {json.dumps(results, indent=2)}",
            },
        ],
    )

    return {
        "final_response": response.choices[0].message.content,
        "agents_involved": len(results),
        "task": task,
    }


if __name__ == "__main__":
    # Test the multi-agent system
    tasks = [
        "Write a brief market analysis for AI observability tools",
        "Create a summary of best practices for LLM cost optimization",
    ]

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print("=" * 60)

        result = orchestrator_agent(task)
        print(f"\nFinal Response:\n{result['final_response']}")
        print(f"\nAgents involved: {result['agents_involved']}")

    LLMObs.disable()
