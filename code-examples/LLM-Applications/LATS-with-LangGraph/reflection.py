from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field


class Reflection(BaseModel):
    """
    Represents an evaluation of a candidate response.
    Includes textual reflection, numerical score, and solution status.
    Provides methods for message formatting and score normalization.
    """

    reflections: str = Field(
        description="The critique and reflections on the sufficiency, superfluency,"
        " and general quality of the response"
    )
    score: int = Field(
        description="""
        Score from 0-10 on the quality of the candidate response. 
        0 means that the response is completely unrelated to the task.
        10 means that the response completely solves the task.
        """,
        gte=0,
        lte=10,
    )
    found_solution: bool = Field(
        description="Whether the response has fully solved the question or task.",
    )

    def as_message(self):
        return HumanMessage(
            content=f"Reasoning: {self.reflections}\nScore: {self.score}"
        )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0
