from typing import List
from pydantic import BaseModel

class DocsResponse(BaseModel):
    answer: str
    sources: List[str]
