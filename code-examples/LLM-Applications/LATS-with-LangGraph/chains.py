from langchain_openai import ChatOpenAI
from prompts import answer_prompt, reflection_prompt
from tools import tavily_tool
from reflection import Reflection
from langchain_core.runnables import chain
from langchain_core.messages import AIMessage

llm = ChatOpenAI(model="gpt-4o-mini")

@chain
def reflection_chain(inputs) -> Reflection:
    chain = reflection_prompt | llm.with_structured_output(Reflection)
    reflection = chain.invoke(inputs)
    if not isinstance(inputs["candidate"][-1], AIMessage):
        reflection.found_solution = False
    return reflection

answer_chain = answer_prompt | llm.bind_tools(tools=[tavily_tool])