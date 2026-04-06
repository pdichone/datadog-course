from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import workflow, retrieval, embedding, llm, task
from openai import OpenAI
from typing import Any, Dict, List, Optional
import os
import chromadb

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
chroma = chromadb.Client()
collection = chroma.get_or_create_collection(
    "docs_rag_pipeline",
    metadata={
        "hnsw:space": "cosine",
    },
)


def setup_sample_data():
    """Load sample documents into the vector store."""
    docs = [
        "Our refund policy allows returns within 30 days of purchase. Full refunds are issued for unopened items.",
        "Enterprise pricing starts at $10,000 per year for up to 100 users. Volume discounts are available.",
        "To reset your password, click 'Forgot Password' on the login page and follow the email instructions.",
        "Our API rate limits are 1000 requests per minute for standard plans, 10000 for enterprise.",
        "Technical support is available 24/7 for enterprise customers via phone, email, or chat.",
    ]

    # Generate embeddings
    response = client.embeddings.create(model="text-embedding-3-small", input=docs)

    embeddings = [item.embedding for item in response.data]

    # Add to collection
    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(docs))],
    )
    print(f"Loaded {len(docs)} documents")


@workflow
def rag_pipeline(query: str, user_id: str = None) -> Dict[str, Any]:
    """
    Complete RAG pipeline with comprehensive tracing.

    Args:
        query: User's question
        user_id: Optional user identifier

    Returns:
        Dict containing answer and metadata
    """
    # Annotate workflow with context
    LLMObs.annotate(
        input_data=query, metadata={"user_id": user_id, "pipeline_version": "v1.0"}
    )

    # Step 1: Embed query
    query_embedding = embed_query(query)

    # Step 2: Retrieve documents
    retrieved = retrieve_documents(query_embedding, top_k=3)

    # Step 3: Check relevance
    if not retrieved["documents"]:
        return {
            "answer": "I don't have enough information to answer that question.",
            "sources": [],
            "confidence": 0.0,
        }

    # Step 4: Assemble context
    context = assemble_context(retrieved["documents"])

    # Step 5: Generate response
    answer = generate_response(query, context)

    # Step 6: Calculate confidence
    confidence = calculate_confidence(retrieved["distances"])

    result = {
        "answer": answer,
        "sources": retrieved["documents"],
        "confidence": confidence,
    }

    # Annotate final output
    LLMObs.annotate(
        output_data=result,
        metadata={
            "sources_count": len(retrieved["documents"]),
            "confidence": confidence,
        },
    )

    return result


@embedding
def embed_query(query: str) -> List[float]:
    """
    Generate embedding for user query.
    """
    response = client.embeddings.create(model="text-embedding-3-small", input=query)

    embedding = response.data[0].embedding

    LLMObs.annotate(
        input_data=query,
        metadata={"model": "text-embedding-3-small", "dimensions": len(embedding)},
    )

    return embedding


@retrieval
def retrieve_documents(embedding: List[float], top_k: int = 5) -> Dict[str, Any]:
    """
    Retrieve relevant documents from vector store.
    """
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    documents = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []

    LLMObs.annotate(
        metadata={
            "top_k": top_k,
            "documents_returned": len(documents),
            "min_distance": min(distances) if distances else None,
            "max_distance": max(distances) if distances else None,
        }
    )

    return {"documents": documents, "distances": distances}


@task
def assemble_context(documents: List[str]) -> str:
    """
    Assemble retrieved documents into context string.
    """
    context = "\n\n---\n\n".join(documents)

    LLMObs.annotate(
        metadata={
            "context_length_chars": len(context),
            "document_count": len(documents),
            "avg_doc_length": len(context) / len(documents) if documents else 0,
        }
    )

    return context


@llm(model_name="gpt-4o-mini", model_provider="openai")
def generate_response(query: str, context: str) -> str:
    """
    Generate answer based on retrieved context.
    """
    system_prompt = f"""You are a helpful customer support assistant.
Answer the user's question based ONLY on the following context.
If the context doesn't contain relevant information, say so.

Context:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        max_tokens=300,
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    LLMObs.annotate(
        input_data={"query": query, "context_preview": context[:500]},
        output_data=answer,
        metadata={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    )

    return answer


@task
def calculate_confidence(distances: List[float]) -> float:
    """
    Calculate confidence score based on retrieval distances.
    Lower distance = higher confidence.
    """
    if not distances:
        return 0.0

    # Convert distance to similarity (assuming cosine distance)
    avg_distance = sum(distances) / len(distances)
    confidence = max(0.0, min(1.0, 1 - avg_distance))

    return round(confidence, 3)


if __name__ == "__main__":
    # Setup sample data
    setup_sample_data()

    # Test queries
    queries = [
        "What is your refund policy?",
        "How much does enterprise pricing cost?",
        "How do I reset my password?",
        "What are the API rate limits?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = rag_pipeline(query, user_id="demo_user")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")

    LLMObs.disable()
