"""
Multi-step RAG chat agent using LangGraph for orchestration.

This module defines a sophisticated conversational AI agent that:
1. Routes queries between simple responses and knowledge-enhanced responses
2. Performs intelligent document retrieval and filtering
3. Generates responses with factual grounding and citations
4. Evaluates response quality and iterates for improvement
5. Provides graceful fallback handling

The agent uses a state machine approach with conditional routing
to handle various conversation scenarios robustly.
"""

from dataclasses import dataclass
from typing import Literal
from langgraph.graph import END, StateGraph, START
from app.chat.chat_agent.state import ChatAgentState

from app.chat.chat_agent.nodes.intent_router import intent_router
from app.chat.chat_agent.nodes.retriever import retriever
from app.chat.chat_agent.nodes.generator import generator
from app.chat.chat_agent.nodes.simple_assistant import simple_assistant
from app.chat.chat_agent.nodes.generation_evaluator import generation_evaluator
from app.chat.chat_agent.nodes.fallback import fallback


@dataclass(frozen=True)
class Nodes:
    """Node name constants for the chat agent graph."""
    INTENT_ROUTER = "intent_router"
    RETRIEVER = "retriever"
    GENERATOR = "generator"
    SIMPLE_ASSISTANT = "simple_assistant"
    GENERATION_EVALUATOR = "generation_evaluator"
    FALLBACK = "fallback"


def answer_type_router(state: ChatAgentState):
    """Route to RAG pipeline or simple assistant based on intent analysis."""
    if state.need_rag:
        return Nodes.RETRIEVER
    else:
        return Nodes.SIMPLE_ASSISTANT
    

def empty_document_router(state: ChatAgentState):
    """Handle cases where retrieval returns no relevant documents."""
    if state.retrieved_documents:
        return Nodes.GENERATOR
    elif state.retrieval_iterations <= 3:
        # Retry retrieval with different routing
        return Nodes.INTENT_ROUTER
    else:
        return Nodes.FALLBACK
    

def generation_evaluation_router(state: ChatAgentState):
    """Route based on response quality evaluation results."""
    if state.generation_evaluation.is_grounded and state.generation_evaluation.is_valid:
        return END
    elif state.generation_iterations <= 3:
        # Regenerate with feedback
        return Nodes.GENERATOR
    else: 
        return Nodes.FALLBACK


# Build the agent graph with nodes and routing logic
builder = StateGraph(ChatAgentState)

# Add all processing nodes
builder.add_node(Nodes.INTENT_ROUTER, intent_router)
builder.add_node(Nodes.RETRIEVER, retriever)
builder.add_node(Nodes.GENERATOR, generator)
builder.add_node(Nodes.SIMPLE_ASSISTANT, simple_assistant)
builder.add_node(Nodes.GENERATION_EVALUATOR, generation_evaluator)
builder.add_node(Nodes.FALLBACK, fallback)

# Define the conversation flow
builder.add_edge(START, Nodes.INTENT_ROUTER)

# Route based on whether RAG is needed
builder.add_conditional_edges(
    Nodes.INTENT_ROUTER, 
    answer_type_router,
    {
        Nodes.RETRIEVER: Nodes.RETRIEVER,
        Nodes.SIMPLE_ASSISTANT: Nodes.SIMPLE_ASSISTANT
    }
)

# Handle retrieval outcomes
builder.add_conditional_edges(
    Nodes.RETRIEVER, 
    empty_document_router,
    {
        Nodes.GENERATOR: Nodes.GENERATOR,
        Nodes.INTENT_ROUTER: Nodes.INTENT_ROUTER,  # Retry with different routing
        Nodes.FALLBACK: Nodes.FALLBACK
    }
)

# RAG pipeline: retrieval -> generation -> evaluation
builder.add_edge(Nodes.RETRIEVER, Nodes.GENERATOR) 
builder.add_edge(Nodes.GENERATOR, Nodes.GENERATION_EVALUATOR)

# Handle generation quality evaluation
builder.add_conditional_edges(
    Nodes.GENERATION_EVALUATOR, 
    generation_evaluation_router,
    {
        END: END,
        Nodes.GENERATOR: Nodes.GENERATOR,  # Regenerate with feedback
        Nodes.FALLBACK: Nodes.FALLBACK
    }
)

# Terminal nodes
builder.add_edge(Nodes.FALLBACK, END) 
builder.add_edge(Nodes.SIMPLE_ASSISTANT, END) 

# Compile the agent for execution
chat_agent = builder.compile()