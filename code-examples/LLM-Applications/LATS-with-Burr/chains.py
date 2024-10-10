from openai import OpenAI
from prompts import answer_system_prompt, reflection_system_prompt
from tools import tavily_client
from langchain_core.utils.function_calling import convert_to_openai_function
from reflection import Reflection
import os


client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

model = "gpt-4o-mini"


def answer_chain(input, messages=None):

    all_messages = [answer_system_prompt, {'content': input, 'role': 'user'}]
    if messages:
        all_messages.extend(messages)

    tools = [{
        "type": "function", 
        "function": convert_to_openai_function(tavily_client.search)
    }]

    response = client.chat.completions.create(
        model=model,
        messages=all_messages,
        tools=tools,
    )

    return response.choices[0].message


def reflection_chain(input, candidate) -> Reflection:
    messages = [reflection_system_prompt, {'content': input, 'role': 'user'}]
    messages.extend(candidate)

    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=Reflection,
    )

    reflection = response.choices[0].message.parsed

    if candidate[-1]['role'] != 'assistant':
        reflection.found_solution = False

    return reflection



    

