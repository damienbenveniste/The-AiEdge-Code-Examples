import os
from langchain_community.tools.tavily_search import TavilySearchResults

os.environ['TAVILY_API_KEY'] = 'YOUR API KEY'

web_search_tool = TavilySearchResults(k=3)