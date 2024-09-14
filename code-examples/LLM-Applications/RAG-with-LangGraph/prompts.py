from langchain_core.prompts import ChatPromptTemplate


system_prompt = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Additional feedback may be provided about a previous version of the answer. Make sure to utilize that feedback to improve the answer.
Only provide the answer and nothing else!
"""

human_prompt = """
Question: {question}

Context: 
{context}

Here is the feedback about previous versions of the answer:
{feedback}

Answer:
"""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
)

system_prompt = """
You a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval.
The vectorstore contains the whole GitHub repository of the LangChain Python Package. Look at the input and try to reason about the underlying semantic intent / meaning.
Additional feedback may be provided for why a previous version of the question didn't lead to a valid response. Make sure to utilize that feedback to generate a better question.
Only respond with the rewritten question and nothing else! 
"""

human_prompt = """
Here is the initial question: {question}

Here is the feedback about previous versions of the question:
{feedback}

Formulate an improved question.
Rewritten question:
"""

db_query_rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
)

system_prompt = """
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by the set of facts.
"""

human_prompt = """
Set of facts:

{documents}

LLM generation: {generation}
"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
) 

system_prompt = """
You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question.
"""

human_prompt = """
User question: {question} 

LLM generation: {generation}
"""


answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
)

system_prompt = """
Your role is to give feedback on a the LLM generated answer. The LLM generation is NOT grounded in the set of retrieved facts.
Explain how the generated answer could be improved so that it is only solely grounded in the retrieved facts.  
Only provide your feedback and nothing else!
"""

human_prompt = """
User question: {question}

Retrieved facts: 
{documents}

Wrong generated answer: {generation}
"""

generation_feedback_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
)

system_prompt = """
Your role is to give feedback on a the text query used to retrieve documents. Those retrieved documents are used as context to answer a user question.
The following generated answer doesn't address the question! Explain how the query could be improved so that the retrieved documents could be more relevant to the question. 
Only provide your feedback and nothing else!
"""

human_prompt = """
User question: {question}

Text query: {rewritten_question}

Retrieved documents: 
{documents}

Wrong generated answer: {generation}
"""

query_feedback_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
)

system_prompt = """
You job is to generate an apology for not being able to provide a correct answer to a user question.
The question were used to retrieve documents from a database and a websearch and none of them were able to provide enough context to answer the user question.
Explain to the user that you couldn't answer the question.
"""

give_up_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "User question: {question} \n\n Answer:"),
    ]
)

system_prompt = """
You are a grader assessing relevance of a retrieved document to a user question. 
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. 'yes' means that the document contains relevant information.
"""

human_prompt = """
Retrieved document: {document}

User question: {question}
"""

grade_doc_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
)

system_prompt = """
You are a knowledge refinement engine. Your job is to extract the information from a document that could be relevant to a user question. 
The goal is to filter out the noise and keep only the information that can provide context to answer the user question.
If the document contains keyword(s) or semantic meaning related to the user question, consider it as relevant.
DO NOT modify the text, only return the original text that is relevant to the user question. 
"""

human_prompt = """
Retrieved document: {document}

User question: {question}
"""

knowledge_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
)

system_prompt = """
You are an expert at routing a user question to a vectorstore, a websearch or a simple QA language model.
The vectorstore contains documents related to Langchain.
If you can answer the question without any additional context or if a websearch could not provide additional context, route it to the QA language model.
If you need additional context and it is a question about Langchain, use the vectorstore, otherwise, use websearch.
"""

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

system_prompt = """
You are a question re-writer that converts an input question to a better version that is optimized for web search. 
Look at the input and try to reason about the underlying semantic intent / meaning.
Additional feedback may be provided for why a previous version of the question didn't lead to a valid response. Make sure to utilize that feedback to generate a better question.
Only respond with the rewritten question and nothing else! 
"""

human_prompt = """
Here is the initial question: {question}

Here is the feedback about previous versions of the question:
{feedback}

Formulate an improved question.
Rewritten question:
"""

websearch_query_rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
)

system_prompt = """
You are a helpful assistant. Provide a answer to the user.
"""

simple_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)