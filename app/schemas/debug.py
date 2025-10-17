from pydantic import BaseModel, Field
from typing import List


class HealthResponse(BaseModel):  
    status: str = Field(..., description="Health status of the application")

class ModelInfo(BaseModel):
    number_of_labels: int = Field(..., description="Number of classification labels")
    model_last_updated: str = Field(..., description="Timestamp of last model update")

class Message(BaseModel):
    role: str = Field(..., description="Message role: system | user | assistant")
    content: str = Field(..., description="Message content text")

class SessionStateSchema(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    history: List[Message] = Field(default_factory=list, description="Conversation history")
    predictions: List[str] = Field(default_factory=list, description="Classifier predictions")