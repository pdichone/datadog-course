import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ddtrace.llmobs import LLMObs
from dotenv import load_dotenv

load_dotenv()

app_name = os.getenv("DD_LLMOBS_ML_APP", "my-llm-app")


LLMObs.enable(
    ml_app=app_name,
    agentless_enabled=True,
    api_key=os.getenv("DD_API_KEY"),
    site=os.getenv("DD_SITE", "datadoghq.com"),
)

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Create vector store
texts = ["Your documents here..."]
vectorstore = Chroma.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create RAG chain using LCEL (modern approach)
prompt = ChatPromptTemplate.from_template(
    "Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {question}"
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# This call is automatically traced!
result = chain.invoke("What is the refund policy?")
print(result)
