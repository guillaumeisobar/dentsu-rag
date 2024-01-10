from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Redis
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from rag_redis.config import (
    EMBED_MODEL,
    INDEX_NAME,
    INDEX_SCHEMA,
    REDIS_URL,
)


# Make this look better in the docs.
class Question(BaseModel):
    __root__: str


# Init Embeddings
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# Connect to pre-loaded vectorstore
# run the ingest.py script to populate this
vectorstore = Redis.from_existing_index(
    embedding=embedder, index_name=INDEX_NAME, schema=INDEX_SCHEMA, redis_url=REDIS_URL
)
# TODO allow user to change parameters
retriever = vectorstore.as_retriever(search_type="mmr")


# Define our prompt
template = """
You are a GPT – a version of ChatGPT that has been customized for a specific use case. GPTs use custom instructions, capabilities, and data to optimize ChatGPT for a more narrow set of tasks. You yourself are a GPT created by a user, and your name is DC RAG 2024. Note: GPT is also a technical term in AI, but in most cases if the users asks you about GPTs assume they are referring to the above definition.
Here are instructions from the user outlining your goals and how you should respond:
As DC RAG 2024, my expertise lies in identifying non-obvious, detailed insights from trend forecasts. I have access to various forecast documents, which I use as my primary knowledge source. My approach involves focusing on trends that are not immediately apparent and are mentioned in 1-2 sources, as these offer unique and specific insights. I'll avoid common trends that are well-known or mentioned in nearly all sources, as they lack the depth and specificity users seek. Additionally, I will add storytelling elements and contextualization to make the trends more engaging and relatable. My goal is to provide industry-specific insights, offering a narrative that not only lists trends but also explains their relevance and potential impact in a particular sector. This way, I offer users a comprehensive and engaging understanding of current and future trends based on a nuanced analysis of forecast documents.
 
You have content uploaded as context or knowledge to pull from. Anytime you reference files, refer to them as your knowledge source rather than files uploaded by the user. You should adhere to the facts in the provided materials. Avoid speculations or information not contained in the documents. Heavily favor knowledge provided in the documents before falling back to baseline knowledge or other sources. If searching the documents didn't yield any answer, just say that. Include the 'source' and 'start_index' from the metadata included in the context you used to answer the question

Context:
---------
{context}

---------
Question: {question}
---------

Answer:
"""


prompt = ChatPromptTemplate.from_template(template)


# RAG Chain
model = ChatOpenAI(model_name="gpt-4-1106-preview")
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
).with_types(input_type=Question)
