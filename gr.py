from dotenv import load_dotenv , find_dotenv
load_dotenv(find_dotenv())


import os
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

ODOO_URL = os.environ["ODOO_URL"]
ODOO_DB = os.environ["ODOO_DB"]
ODOO_USERNAME = os.environ["ODOO_USERNAME"]
ODOO_API = os.environ["ODOO_API"]

# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.document_loaders import TextLoader

from langchain.document_loaders.csv_loader import CSVLoader


import weaviate
import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI 
from langchain.llms import Cohere
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage ,AIMessage
from langchain.agents import Tool , AgentType,initialize_agent
from langchain.tools import StructuredTool
from langchain.prompts import MessagesPlaceholder
from langchain.agents.agent_toolkits import create_retriever_tool ,create_conversational_retrieval_agent
from langchain.chat_models import JinaChat

import gradio as gr

from odoo_functions import odoo

from pydantic import BaseModel , Field

import agent_prompt



odoo_client = odoo(ODOO_URL,ODOO_DB,ODOO_USERNAME,ODOO_API)


langchain.debug = True
langchain.verbose = True




client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY),additional_headers={
        "X-Cohere-Api-Key": COHERE_API_KEY, # Replace with your cohere key
        })


loader = CSVLoader(file_path='./data/products.csv')
docs = loader.load()

embeddings = CohereEmbeddings()

db = Weaviate.from_documents(docs, embeddings, client=client, by_text=False)


from langchain.utilities import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()


tools =[
    # create_retriever_tool(
    # db.as_retriever(), 
    # "search_nearest_product_name",
    # "Searches and returns products names near product needed to use before search_products"
    # ),
    Tool.from_function(
        odoo_client.search_products,
        "search_store",
        "useful when search for products services or food at Ama ecommerce store to get price product_name must be one of provided products",
        # args_schema=SearchProductInput
    ),
    StructuredTool.from_function(
        odoo_client.place_order,
        "place_order",
        "useful when place order at Ama ecommerce store",
        # args_schema=PlaceOrderInput
    ),
    Tool(
    name="Search useful product",
    description="Search for product that can help solve customer problem",
    func=search.run,
    )
    ]

chat_history = MessagesPlaceholder(variable_name="chat_history")


cohere_llm = Cohere(model="command")
jina_chat = JinaChat(temperature=0)
openai_chat = ChatOpenAI(temperature=0, model="gpt-4")

# memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(tools,openai_chat, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True ,agent_kwargs={
        'prefix': agent_prompt.PREFIX, 
        'format_instructions': agent_prompt.FORMAT_INSTRUCTIONS,
        'suffix': agent_prompt.SUFFIX,
        "memory_prompts": [chat_history],
        "input_variables": ["input", "agent_scratchpad", "chat_history" ,"products"]
    })#, memory=memory)



# print(odoo_client.search_products("burger"))


def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        # history_langchain_format.append((human, ai))
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    # history_langchain_format.append(HumanMessage(content=message))
    # gpt_response = llm(history_langchain_format)
    # return chain( {"question": message , "chat_history":history_langchain_format}, return_only_outputs=True )['answer']

    # ag_output = agent( {"input": message  , "chat_history": history_langchain_format} )
    products = db.similarity_search(message , 10)
    products = [p.page_content for p in products]
    print("********************")
    print(products)
    ag_output = agent.run( input=message  , chat_history=history_langchain_format ,products=" ".join(products))

    print(ag_output)
    # return ag_output['output']
    return ag_output




gr.ChatInterface(predict).queue().launch(server_name="0.0.0.0", server_port=7860)
