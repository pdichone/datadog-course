"""
LangChain RAG with Automatic Instrumentation
Demonstrates zero-code tracing with LangChain
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ddtrace.llmobs import LLMObs
import os

load_dotenv()

app_name = os.getenv("DD_LLMOBS_ML_APP", "my-llm-app")


LLMObs.enable(
    ml_app=app_name,
    agentless_enabled=True,
    api_key=os.getenv("DD_API_KEY"),
    site=os.getenv("DD_SITE", "datadoghq.com"),
)

# embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

"""
LangChain RAG with Automatic Instrumentation
Demonstrates zero-code tracing with LangChain
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ddtrace.llmobs import LLMObs
import os

from dotenv import load_dotenv

load_dotenv()

# Initialize LLM Observability
LLMObs.enable(ml_app="langchain-rag-demo", agentless_enabled=True)


def create_vector_store():
    """Create and populate a Chroma vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Sample enterprise documents
    documents = [
        "Our enterprise SLA guarantees 99.9% uptime with 24/7 support.",
        "Enterprise customers receive dedicated account managers.",
        "API rate limits for enterprise: 10,000 requests per minute.",
        "Data is encrypted at rest using AES-256 encryption.",
        "We are SOC 2 Type II and GDPR compliant.",
        "Enterprise pricing starts at $10,000/year for up to 100 users.",
    ]

    vectorstore = Chroma.from_texts(
        texts=documents, embedding=embeddings, collection_name="enterprise_docs"
    )

    return vectorstore


def create_rag_chain(vectorstore):
    """Create a RAG chain with custom prompt."""
    # llm = ChatOpenAI(model="gpt-4", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Custom prompt
    template = """You are an enterprise sales assistant.
Answer questions about our enterprise offerings based on the context.
Be professional and concise.

Context: {context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Modern LCEL chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def main():
    # Setup
    vectorstore = create_vector_store()
    chain = create_rag_chain(vectorstore)

    # Test queries
    queries = [
        "What is your uptime guarantee?",
        "Tell me about enterprise security features",
        "How much does enterprise cost?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        # This entire chain execution is automatically traced
        response = chain.invoke(query)
        print(f"Response: {response}")

    LLMObs.disable()


if __name__ == "__main__":
    main()
