
from typing import Dict, Any, List
from langchain_core.callbacks import BaseCallbackHandler
import schemas
import crud


class LogResponseCallback(BaseCallbackHandler):

    def __init__(self, user_request: schemas.UserRequest, db):
        super().__init__()
        self.user_request = user_request
        self.db = db

    def on_llm_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        message = schemas.MessageBase(message=outputs.generations[0][0].text, type='AI')
        crud.add_message(self.db, message, self.user_request.username)

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        for prompt in prompts:
            print(prompt)
