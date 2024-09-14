from langgraph.graph import END, StateGraph, START
from nodes import (
    GraphState,
    db_query_rewriting_node,
    retriever_node,
    generation_node,
    router_node,
    query_feedback_node,
    generation_feedback_node,
    simple_question_node,
    web_search_node,
    websearch_query_rewriting_node,
    give_up_node,
    filter_relevant_documents_node,
    knowledge_extractor_node,
    answer_evaluation_node,
    search_mode_node,
    relevant_documents_validation_node
)


pipeline = StateGraph(GraphState)

pipeline.add_node('db_query_rewrite_node', db_query_rewriting_node)
pipeline.add_node('retrieval_node', retriever_node)
pipeline.add_node('generator_node', generation_node)
pipeline.add_node('query_feedback_node', query_feedback_node)
pipeline.add_node('generation_feedback_node', generation_feedback_node)
pipeline.add_node('simple_question_node', simple_question_node)
pipeline.add_node('websearch_query_rewriting_node', websearch_query_rewriting_node)
pipeline.add_node('web_search_node', web_search_node)
pipeline.add_node('give_up_node', give_up_node)
pipeline.add_node('filter_docs_node', filter_relevant_documents_node)
pipeline.add_node('extract_knowledge_node', knowledge_extractor_node)

pipeline.add_conditional_edges(
    START, 
    router_node,
    {
        "vectorstore": 'db_query_rewrite_node',
        "websearch": 'websearch_query_rewriting_node',
        "QA_LM": 'simple_question_node'
    },
)

pipeline.add_edge('db_query_rewrite_node', 'retrieval_node')
pipeline.add_edge('retrieval_node', 'filter_docs_node')
pipeline.add_edge('extract_knowledge_node', 'generator_node')
pipeline.add_edge('websearch_query_rewriting_node', 'web_search_node')
pipeline.add_edge('web_search_node', 'filter_docs_node')
pipeline.add_edge('generation_feedback_node', 'generator_node')
pipeline.add_edge('simple_question_node', END)
pipeline.add_edge('give_up_node', END)

pipeline.add_conditional_edges(
    'generator_node', 
    answer_evaluation_node,
    {
        "useful": END,
        "not relevant": 'query_feedback_node',
        "hallucination": 'generation_feedback_node',
        "max_generation_reached": 'give_up_node'
    }  
)

pipeline.add_conditional_edges(
    'query_feedback_node', 
    search_mode_node,
    {
        "vectorstore": 'db_query_rewrite_node',
        "websearch": 'websearch_query_rewriting_node',
    }
)

pipeline.add_conditional_edges(
    'filter_docs_node', 
    relevant_documents_validation_node,
    {
        "knowledge_extraction": 'extract_knowledge_node',
        "websearch": 'websearch_query_rewriting_node',
        "vectorstore": 'db_query_rewrite_node',
        "max_db_search": 'websearch_query_rewriting_node',
        "max_websearch": 'give_up_node'
    }
)

rag_pipeline = pipeline.compile()