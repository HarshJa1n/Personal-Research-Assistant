from crewai_tools import SerperDevTool, ScrapeWebsiteTool, LlamaIndexTool
from llama_index.readers.papers import ArxivReader
from llama_index.core import VectorStoreIndex


class Tools:
    
    def __init__(self, user_input):
        self.user_input = user_input
    
    def arxiv_query_tool(self, embed_model, llm):
        loader = ArxivReader()
        documents = loader.load_data(search_query="abs:{}".format(self.user_input))
        index = VectorStoreIndex.from_documents(documents, show_progress=True, embed_model=embed_model)
        query_engine = index.as_query_engine(similarity_top_k=5, llm=llm)
        query_tool = LlamaIndexTool.from_query_engine(
            query_engine,
            name="Arxiv Research Tool",
            description="Use this tool to lookup Arxiv website to look for recent publications on a given topic"
        )
        return query_tool
    
    def serper_dev_tool(self):
        return SerperDevTool()
    
    def scrape_website_tool(self):
        return ScrapeWebsiteTool()