
from tavily import TavilyClient
import os
import json

tavily_client = TavilyClient(api_key=os.environ['TAVILY_API_KEY'])

def search(openai_response):
    if not openai_response.tool_calls:
        return None
    tool_responses = [] 
    for tool_call in openai_response.tool_calls:
        args = json.loads(tool_call.function.arguments)
        tool_response = tavily_client.search(**args)
        tool_response = [
            {'url': res['url'], 'content': res['content']}
            for res in tool_response['results']
        ]
        tool_responses.append((tool_response, tool_call.id))
    return tool_responses