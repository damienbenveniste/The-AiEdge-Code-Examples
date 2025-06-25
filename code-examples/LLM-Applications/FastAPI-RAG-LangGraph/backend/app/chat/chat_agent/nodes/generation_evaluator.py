from app.chat.chat_agent.state import ChatAgentState, EvaluationResult
from app.openai_connect import async_openai_client
import logging


SYSTEM_PROMPT = """
You are an evaluation assistant.

Inputs (JSON) you will receive  
• **chat_history** - the full ordered list of prior messages.  
• **documents** - the exact set of retrieved web-page chunks (`url`, `content`) that the generator saw.  
• **answer** - the assistant reply shown to the user, including any URL citations.

Your tasks  
1. **Hallucination check** - Verify that every explicit factual claim in *answer* is supported by at least one of the provided *documents*.  
2. **Answer validity** - Judge whether *answer* fully and correctly addresses the user’s latest question, given *chat_history*.  

Produce:  
• is_grounded  - true only if no unsupported claims are found.
• is_valid     - true only if the answer is relevant, complete, and correct.
• feedback     - a **single list** that merges *both* kinds of issues **as
  actionable suggestions the generator can follow next round**:
     - For hallucinations → “Remove claim X or cite Y.”
     - For validity gaps   → “Explain Z in more detail,” etc.
  If there are no issues (both booleans are true), set feedback to null.
"""


class GenerationEvaluator:
    """
    Evaluates generated responses for factual accuracy and completeness.
    
    This class performs quality control on AI-generated responses by checking:
    1. Hallucination detection - ensuring claims are supported by retrieved documents
    2. Answer validity - verifying the response addresses the user's question properly
    """

    async def evaluate(self, state: ChatAgentState) -> EvaluationResult:
        """
        Evaluate the quality of a generated response.
        
        Args:
            state (ChatAgentState): Current conversation state with generated response
            
        Returns:
            EvaluationResult: Evaluation with grounding/validity flags and feedback
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"###  Chat_history  ###"},
        ]

        messages.extend(state.chat_messages[-10:])
        documents = '\n'.join([doc.model_dump_json(indent=2) for doc in state.retrieved_documents])
        messages.append({
            "role": "user", 
            "content": f"###  Documents  ###\n\n{documents}"
        })

        messages.append({
            "role": "assistant", 
            "content": state.generation
        })

        try:
            response = await async_openai_client.responses.parse(
                model='gpt-4.1-mini',
                input=messages,
                temperature=0.1,
                text_format=EvaluationResult,
            )
        except Exception as e:
            logging.error(str(e))
            raise ConnectionError("Something wrong with Openai: {e}")

        return response.output_parsed
    
    async def __call__(self, state: ChatAgentState) -> ChatAgentState:
        """
        Execute the generation evaluation step.
        
        Args:
            state (ChatAgentState): Current conversation state
            
        Returns:
            ChatAgentState: Updated state with evaluation results
        """
        generation_evaluation = await self.evaluate(state)
        state.generation_evaluation = generation_evaluation
        return state
    

generation_evaluator = GenerationEvaluator()