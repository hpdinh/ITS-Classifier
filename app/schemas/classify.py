from typing import Optional, List, Annotated

from pydantic import BaseModel, Field

class ClassificationRequest(BaseModel):
    text: str
    top_k: Optional[int] = 3  # default to 3 if not provided

class ClassificationResponse(BaseModel):
    predictions: List[str] = Field(..., description="List of predicted classifications")
