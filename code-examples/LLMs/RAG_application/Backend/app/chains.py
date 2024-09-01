import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough
import schemas
from prompts import (
    standalone_prompt_formatted,
    rag_prompt_formatted,
    format_context,
    tokenizer
)
from data_indexing import DataIndexer

data_indexer = DataIndexer()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    huggingfacehub_api_token=os.environ['HF_TOKEN'],
    max_new_tokens=512,
    stop_sequences=[tokenizer.eos_token],
    streaming=True,
)

rag_chain = (
    RunnablePassthrough.assign(new_question=standalone_prompt_formatted | llm)
    | {
        'context': lambda x: format_context(data_indexer.search(x['new_question'], hybrid_search=x['hybrid_search'])),
        'standalone_question': lambda x: x['new_question'],
    }
    | rag_prompt_formatted
    | llm
).with_types(input_type=schemas.RagInput)





