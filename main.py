import os
import warnings
import requests
from dotenv import load_dotenv

import streamlit as st

from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter

from crewai import Crew, Process
from agents import *
from tools import *
from tasks import *

warnings.filterwarnings("ignore")
load_dotenv()

st.title("Literature Survey Made Easy!")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = st.text_input("Enter your OpenAI API key: ", type="password")

LLM_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL_NAME, api_key=OPENAI_API_KEY)
llm = OpenAI(model=LLM_MODEL_NAME, api_key=OPENAI_API_KEY)

if OPENAI_API_KEY:
    try:
        choice = st.radio("Select the task you are interested in", ["Interact with a Research Paper", "Research about a research topic"])
        if choice == "Interact with a Research Paper":
            pdf_url = st.text_input("Enter PDF url")
            if pdf_url:
                try:
                    response = requests.get(pdf_url)
                    save_path = "paper.pdf"
                    with open(save_path, 'wb') as file:
                        file.write(response.content)
                    reader = SimpleDirectoryReader(input_files=[save_path])
                    documents = reader.load_data()
                    
                    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
                    nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)
                    index = VectorStoreIndex.from_documents(documents, 
                                                            show_progress=True, 
                                                            embed_model=embed_model, 
                                                            node_parser=nodes)
                    query_engine = index.as_query_engine(similarity_top_k=5, llm=llm)
                    question = st.text_input("Ask question", key="question")

                    if st.button("Submit", key="5"):
                        if question:
                            st.write("Searching.......")
                            response = query_engine.query(question)
                            st.write(response.response)
                        else:
                            st.write("Ask question")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                if not pdf_url:
                    st.write("Please provide the url to download the PDF.")
        else:   
            user_input = st.text_input("Which research topic you are interested in?")

            if st.button("Submit", key="3"):
                if user_input:
                    st.write("please have patience, Our Crew is at work :)")

                    tools = Tools(user_input)
                    serper_dev_tool = tools.serper_dev_tool()
                    scrape_website_tool = tools.scrape_website_tool()
                    arxiv_query_tool = tools.arxiv_query_tool(embed_model, llm)
                    
                    agents = Agents(user_input)
                    web_researcher = agents.web_researcher(serper_dev_tool, scrape_website_tool)
                    arxiv_researcher = agents.arxiv_researcher(arxiv_query_tool)
                    analyst = agents.analyst()
                    manager = agents.manager()

                    tasks = Tasks(user_input)
                    web_researcher_task = tasks.web_researcher_task(web_researcher)
                    arxiv_researcher_task = tasks.arxiv_researcher_task(arxiv_researcher)
                    analyst_task = tasks.analyst_task(analyst)

                    crew = Crew(
                        agents=[web_researcher, arxiv_researcher, analyst],
                        tasks=[web_researcher_task, arxiv_researcher_task, analyst_task],
                        manager_agent=manager,
                        process=Process.hierarchical,
                        # cache=True,
                        # memory=True,
                        verbose=True,
                    )
                    result = crew.kickoff()
                    st.write("AI response: ")
                    st.write(result.raw)
                else:
                    st.write("Please enter research topic")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    if not OPENAI_API_KEY:
        st.write("Please provide your OpenAI API key")