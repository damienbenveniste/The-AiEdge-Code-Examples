from transformers import AutoTokenizer
from langchain_core.prompts import PromptTemplate
from typing import List
import models

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

standalone_prompt = """
Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question:
"""

rag_prompt = """
Answer the question based only on the following context:
{context}

Question: {standalone_question}
"""


def format_prompt(prompt):
    chat = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        chat, 
        tokenize=False, 
        add_generation_prompt=True
    )

    return PromptTemplate.from_template(formatted_prompt)


def format_chat_history(messages: List[models.Message]):
    return '\n'.join([
        '{}: {}'.format(message.type, message.message)
        for message in messages
    ])

def format_context(docs: List[str]):
    return '\n\n'.join(docs)


standalone_prompt_formatted = format_prompt(standalone_prompt)
rag_prompt_formatted = format_prompt(rag_prompt)

