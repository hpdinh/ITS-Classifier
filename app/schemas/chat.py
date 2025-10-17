from typing import List, Literal, Optional
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: Literal["system","user","assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    k: Optional[int] = 3  # how many Confluence snippets to include

class ChatResponse(BaseModel):
    reply: str