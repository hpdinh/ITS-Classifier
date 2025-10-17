# core/dependencies.py

"""
Dependency Providers for FastAPI.

This module defines small helper functions that retrieve shared services
from the FastAPI application state or construct session objects. These
functions are intended to be used with FastAPI's dependency injection
mechanism (`Depends`) in route handlers.

Provided dependencies:
    - ClassifierService
    - LLMClient
    - SessionState
    - SnowTicketQuery (RAG: tickets)
    - SnowAssignmentGroupsRetriever (RAG: Confluence)
"""

import uuid
from fastapi import Request
from services.classifier_service import ClassifierService
from services.llm_client import LLMClient
from services.session_service import SessionState
from services.rag.ticket_query import SnowTicketQuery
from services.rag.groups_query import SnowAssignmentGroupsRetriever


# ---- Classifier ----
def get_classifier(request: Request) -> ClassifierService:
    """
    Retrieve the shared classifier service from app state.

    Args:
        request (Request): FastAPI request object.

    Returns:
        ClassifierService: The classifier instance.
    """
    return request.app.state.classifier


# ---- LLM ----
def get_llm_client(request: Request) -> LLMClient:
    """
    Retrieve the shared LLM client from app state.

    Args:
        request (Request): FastAPI request object.

    Returns:
        LLMClient: The LLM client instance.
    """
    return request.app.state.llm_client


# ---- Session ----
def get_session(request: Request) -> SessionState:
    """
    Load or create a session state object.

    If the request contains a `session_id` cookie, load the corresponding
    session. Otherwise, generate a new session ID and start a new session.

    Args:
        request (Request): FastAPI request object.

    Returns:
        SessionState: Session state object, persisted via Redis or in-memory.
    """
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())   # generate new if missing
    session = SessionState.load(session_id)
    session.session_id = session_id
    return session


# ---- RAG: Tickets ----
def get_ticket_query(request: Request) -> SnowTicketQuery:
    """
    Retrieve the ticket query service (Qdrant retriever) from app state.

    Args:
        request (Request): FastAPI request object.

    Returns:
        SnowTicketQuery: Ticket retriever instance for dense/sparse search.
    """
    return request.app.state.ticket_query


# ---- RAG: Confluence ----
def get_groups_retriever(request: Request) -> SnowAssignmentGroupsRetriever:
    """
    Retrieve the assignment groups retriever (Confluence integration) from app state.

    Args:
        request (Request): FastAPI request object.

    Returns:
        SnowAssignmentGroupsRetriever: Retriever for SNOW assignment group lookups.
    """
    return request.app.state.groups_retriever