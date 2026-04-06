"""
Instrumented LLM Calls
Demonstrates custom span creation and annotation
"""

from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import workflow, retrieval, embedding, llm, task
from openai import OpenAI
from typing import Optional
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
collection = chroma.get_or_create_collection("documents")


def seed_collection():
    """Seed ChromaDB with sample documents if empty."""
    if collection.count() > 0:
        return

    documents = [
        "Green tea contains catechins, powerful antioxidants that help protect cells from damage and reduce inflammation in the body.",
        "Studies show that green tea can boost metabolism and increase fat burning, making it beneficial for weight management.",
        "The L-theanine in green tea promotes relaxation and focus without drowsiness, improving mental clarity and cognitive function.",
        "Regular green tea consumption has been linked to lower risk of heart disease by reducing LDL cholesterol and improving blood vessel function.",
        "Green tea polyphenols may help regulate blood sugar levels, potentially reducing the risk of type 2 diabetes.",
        "Research suggests that the EGCG in green tea has anti-cancer properties and may inhibit tumor growth in various types of cancer.",
        "Green tea supports oral health by inhibiting bacteria growth, reducing bad breath and lowering the risk of cavities.",
        "The antioxidants in green tea may help protect the brain from aging, potentially reducing the risk of Alzheimer's and Parkinson's diseases.",
    ]

    # Generate embeddings for all documents
    response = client.embeddings.create(model="text-embedding-3-small", input=documents)
    embeddings = [item.embedding for item in response.data]

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))],
    )
    print(f"Seeded ChromaDB with {len(documents)} documents.")


@workflow
def rag_pipeline(query: str) -> str:
    """
    Complete RAG pipeline with full tracing.

    This creates a parent span containing all child operations.
    """
    # Step 1: Embed the query
    query_embedding = embed_query(query)

    # Step 2: Retrieve documents
    documents = retrieve_documents(query, query_embedding)

    # Step 3: Assemble context
    context = assemble_context(documents)

    # Step 4: Generate response
    response = generate_response(query, context)

    return response


@embedding
def embed_query(query: str) -> list:
    """Create embedding for the query."""
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    return response.data[0].embedding


@retrieval
def retrieve_documents(query: str, embedding: list, top_k: int = 5) -> list:
    """Retrieve relevant documents from vector store."""
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    documents = results["documents"][0]
    distances = results["distances"][0]

    # Annotate retrieval span with input, output, and metadata
    LLMObs.annotate(
        input_data=query,
        output_data=[
            {"text": doc, "id": f"doc_{i}", "score": 1 - dist}
            for i, (doc, dist) in enumerate(zip(documents, distances))
        ],
        metadata={
            "top_k": top_k,
            "documents_returned": len(documents),
        },
    )

    return documents


@task
def assemble_context(documents: list) -> str:
    """Assemble documents into context string."""
    context = "\n\n---\n\n".join(documents)

    LLMObs.annotate(
        metadata={"context_length": len(context), "document_count": len(documents)}
    )

    return context


@llm(model_name="gpt-4o-mini", model_provider="openai")
def generate_response(query: str, context: str) -> str:
    """Generate answer based on context."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"Answer based on this context:\n\n{context}",
            },
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    seed_collection()
    user_query = "What are the health benefits of green tea?"
    answer = rag_pipeline(user_query)
    print("Answer:", answer)

    LLMObs.disable()
