"""
Query Schema Module.

Defines Pydantic models for handling search queries and responses
from a vector database or retrieval system.

Models:
    - QueryRequest: Incoming search request with query text and top-k parameter.
    - QueryResult: A single result item from the vector store, including ID, score, and metadata.
    - QueryResponse: Wrapper for a list of QueryResult objects.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    """
    Schema for a search query request.

    Attributes:
        query (str): Search query string.
        top_k (Optional[int]): Number of results to return. Defaults to 5.
    """
    query: str = Field(..., description="Search query string")
    top_k: Optional[int] = Field(5, description="Number of results to return")


class QueryResult(BaseModel):
    """
    Schema for a single search result.

    Represents an item retrieved from a vector store or retrieval engine.

    Attributes:
        id (str): Document ID from the vector store.
        score (float): Similarity score of the match.
        payload (Dict[str, Any]): Original document metadata associated with the result.
    """
    id: str = Field(..., description="Document ID from vector store")
    score: float = Field(..., description="Similarity score")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Original document metadata")


class QueryResponse(BaseModel):
    """
    Schema for a search query response.

    Attributes:
        results (List[QueryResult]): List of retrieved documents.
    """
    results: List[QueryResult] = Field(..., description="List of retrieved documents")
