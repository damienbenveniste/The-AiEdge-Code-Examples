from typing import List, Literal, Optional
from langchain_core.pydantic_v1 import BaseModel
from data_index import retriever

from chains import (
    rag_chain,
    db_query_rewriter,
    hallucination_grader,
    answer_grader,
    generation_feedback_chain,
    query_feedback_chain,
    retrieval_grader,
    knowledge_extractor,
    question_router,
    simple_question_chain,
    give_up_chain,
    websearch_query_rewriter
)

from websearch import web_search_tool
from data_index import retriever


MAX_RETRIEVALS = 3
MAX_GENERATIONS = 3

class GraphState(BaseModel):

    question: Optional[str] = None
    generation: Optional[str] = None
    documents: List[str] = []
    rewritten_question: Optional[str] = None
    query_feedbacks: List[str] = []
    generation_feedbacks: List[str] = []
    generation_num: int = 0
    retrieval_num: int = 0
    search_mode: Literal["vectorstore", "websearch", "QA_LM"] = "QA_LM"

def retriever_node(state: GraphState):
    new_documents = retriever.invoke(state.rewritten_question)
    new_documents = [d.page_content for d in new_documents]
    state.documents.extend(new_documents)
    return {
        "documents": state.documents, 
        "retrieval_num": state.retrieval_num + 1
    }

def generation_node(state: GraphState):
    generation = rag_chain.invoke({
        "context": "\n\n".join(state.documents), 
        "question": state.question, 
        "feedback": "\n".join(state.generation_feedbacks)
    })
    return {
        "generation": generation,
        "generation_num": state.generation_num + 1
    }

def db_query_rewriting_node(state: GraphState):
    rewritten_question = db_query_rewriter.invoke({
        "question": state.question,
        "feedback": "\n".join(state.query_feedbacks)
    })
    return {"rewritten_question": rewritten_question, "search_mode": "vectorstore"} 

def answer_evaluation_node(state: GraphState):
    # assess hallucination
    hallucination_grade = hallucination_grader.invoke(
        {"documents": state.documents, "generation": state.generation}
    )
    if hallucination_grade.binary_score == "yes":
        # if no hallucination, assess relevance
        answer_grade = answer_grader.invoke({
            "question": state.question, 
            "generation": state.generation
        })
        if answer_grade.binary_score == "yes":
            # no hallucination and relevant
            return "useful"
        elif state.generation_num > MAX_GENERATIONS:
            return "max_generation_reached"
        else:
            # no hallucination but not relevant
            return "not relevant"
    elif state.generation_num > MAX_GENERATIONS:
        return "max_generation_reached"
    else:
        # we have hallucination
        return "hallucination" 
    
def generation_feedback_node(state: GraphState):
    feedback = generation_feedback_chain.invoke({
        "question": state.question,
        "documents": "\n\n".join(state.documents),
        "generation": state.generation
    })

    feedback = 'Feedback about the answer "{}": {}'.format(
        state.generation, feedback
    )
    state.generation_feedbacks.append(feedback)
    return {"generation_feedbacks": state.generation_feedbacks}

def query_feedback_node(state: GraphState):
    feedback = query_feedback_chain.invoke({
        "question": state.question,
        "rewritten_question": state.rewritten_question,
        "documents": "\n\n".join(state.documents),
        "generation": state.generation
    })

    feedback = 'Feedback about the query "{}": {}'.format(
        state.rewritten_question, feedback
    )
    state.query_feedbacks.append(feedback)
    return {"query_feedbacks": state.query_feedbacks}

def give_up_node(state: GraphState):
    response = give_up_chain.invoke(state.question)
    return {"generation": response}

def filter_relevant_documents_node(state: GraphState):
    # first, we grade every documents
    grades = retrieval_grader.batch([
        {"question": state.question, "document": doc} 
        for doc in state.documents
    ])
    # Then we keep only the documents that were graded as relevant
    filtered_docs = [
        doc for grade, doc 
        in zip(grades, state.documents) 
        if grade.binary_score == 'yes'
    ]

    # If we didn't get any relevant document, let's capture that 
    # as a feedback for the next retrieval iteration
    if not filtered_docs:
        feedback = 'Feedback about the query "{}": did not generate any relevant documents.'.format(
            state.rewritten_question
        )
        state.query_feedbacks.append(feedback)

    return {
        "documents": filtered_docs, 
        "query_feedbacks": state.query_feedbacks
    }

def knowledge_extractor_node(state: GraphState):
    filtered_docs = knowledge_extractor.batch([
        {"question": state.question, "document": doc} 
        for doc in state.documents
    ])
    # we keep only the non empty documents
    filtered_docs = [doc for doc in filtered_docs if doc]
    return {"documents": filtered_docs}

def router_node(state: GraphState):
    route_query = question_router.invoke(state.question)
    return route_query.route

def simple_question_node(state: GraphState):
    answer = simple_question_chain.invoke(state.question)
    return {"generation": answer, "search_mode": "QA_LM"}

def websearch_query_rewriting_node(state: GraphState):
    rewritten_question = websearch_query_rewriter.invoke({
        "question": state.question, 
        "feedback": "\n".join(state.query_feedbacks)
    })
    if state.search_mode != "websearch":
        state.retrieval_num = 0    
    return {
        "rewritten_question": rewritten_question, 
        "search_mode": "websearch",
        "retrieval_num": state.retrieval_num
    }

def web_search_node(state: GraphState):
    new_docs = web_search_tool.invoke(
        {"query": state.rewritten_question}
    )
    web_results = [d["content"] for d in new_docs]
    state.documents.extend(web_results)
    return {
        "documents": state.documents, 
        "retrieval_num": state.retrieval_num + 1
    }

def search_mode_node(state: GraphState):
    return state.search_mode

def relevant_documents_validation_node(state: GraphState):
    if state.documents:
        # we have relevant documents
        return "knowledge_extraction"
    elif state.search_mode == 'vectorsearch' and state.retrieval_num > MAX_RETRIEVALS:
        # we don't have relevant documents
        # and we reached the maximum number of retrievals
        return "max_db_search"
    elif state.search_mode == 'websearch' and state.retrieval_num > MAX_RETRIEVALS:
        # we don't have relevant documents
        # and we reached the maximum number of websearches
        return "max_websearch"
    else:
        # we don't have relevant documents
        # so we retry the search
        return state.search_mode