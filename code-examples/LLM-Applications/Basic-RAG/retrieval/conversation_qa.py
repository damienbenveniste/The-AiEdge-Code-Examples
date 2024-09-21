from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel


SYSTEM_PROMPT = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know. 
DO NOT provide your own answer, just copy from the context provided and return it 

CONTEXT: 
{context}
"""

HUMAN_PROMPT = "QUESTION: {question}"

def format_context(docs):
    return '\n\n'.join([doc.page_content for doc in docs])


class UserInput(BaseModel):
    input: str


class RAGPipeline:
    def __init__(self, retriever) -> None:
        self.retriever = retriever
        model = ChatOpenAI(model='gpt-4o-mini')
        prompt = ChatPromptTemplate.from_messages([
            ('system', SYSTEM_PROMPT),
            ('human', HUMAN_PROMPT)
        ])

        self.rag_chain = (
            {
                'context': lambda x: format_context(self.retriever.invoke(x['input'])),
                'question': lambda x: x['input']
            }
            | prompt 
            | model
        ).with_types(input_type=UserInput)

    def get_chain(self):
        return self.rag_chain