from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import NumberedListOutputParser


PROMPT_LIST = """
Generate a numbered list of 3 hypothetical questions that the following document could answer:

DOCUMENT: {doc}
"""


class DataProcessor:

    def __init__(self) -> None:
        model = ChatOpenAI(model='gpt-4o-mini')
        prompt = ChatPromptTemplate.from_template(PROMPT_LIST)
        self.chain = prompt | model | NumberedListOutputParser()

    def transform(self, docs):
        all_questions = self.chain.batch([
            {'doc': doc.page_content} for doc in docs
        ])

        transformed_docs = []
        for question_list, doc in zip(all_questions, docs):
            for question in question_list:
                transformed_docs.append((question, doc))

        return transformed_docs






        
