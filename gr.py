from dotenv import load_dotenv , find_dotenv
load_dotenv(find_dotenv())


import os
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.document_loaders import TextLoader

import weaviate

import langchain

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI 
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage ,AIMessage


import gradio as gr

langchain.debug = True
langchain.verbose = True

client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY),additional_headers={
        "X-Cohere-Api-Key": COHERE_API_KEY, # Replace with your cohere key
        })


loader = TextLoader("./state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings()
embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)

db = Weaviate.from_documents(docs, embeddings, client=client, by_text=False)


# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
# agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)


chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), chain_type="stuff", retriever=db.as_retriever()
)


# chain( {"question": "What did the president say about Justice Breyer"}, return_only_outputs=True, )


def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    # gpt_response = llm(history_langchain_format)
    return chain( {"question": message}, return_only_outputs=True )['answer']


gr.ChatInterface(predict).queue().launch()
