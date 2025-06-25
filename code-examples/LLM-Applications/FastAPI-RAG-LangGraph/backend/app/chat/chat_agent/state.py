from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class RetrieveDocument(BaseModel):
    text: str
    url: str


class EvaluationResult(BaseModel):
    """
    Combined hallucination + validity evaluation of a generated answer.

    feedback: If either is_grounded or is_valid is False, provide a list of
    **action-oriented suggestions** the generator can apply (e.g.
    “Cite the 2024 revenue figure from https://…”, “Add a definition of MFA”).
    Otherwise set to None.
    """
    is_grounded: bool = Field(..., description="True if all claims are supported by the documents.")
    is_valid: bool = Field(..., description="True if the answer fully addresses the user's question.")
    feedback: Optional[str] = Field(
        None,
        description="Merged list of actionable suggestions when issues exist; null if no issues."
    )


class ChatAgentState(BaseModel):

    chat_messages: List[Dict[str, str]] = []
    generation: Optional[str] = None
    need_rag: bool = False
    query_vector_db: Optional[str] = None
    retrieved_documents: List[RetrieveDocument] = []
    generation: Optional[str] = None
    generation_evaluation: Optional[EvaluationResult] = None
    generation_iterations: int = 0
    retrieval_iterations: int = 0
