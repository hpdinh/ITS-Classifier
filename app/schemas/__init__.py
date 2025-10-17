# schemas/__init__.py
from .classify import ClassificationRequest, ClassificationResponse
from .chat import ChatRequest, ChatResponse
from .query import QueryRequest, QueryResponse, QueryResult
from .debug import HealthResponse, ModelInfo, SessionStateSchema, Message
from .info_form import InfoForm
from .ticket import Ticket

__all__ = [
    # classify
    "ClassificationRequest",
    "ClassificationResponse",
    # chat
    "ChatRequest",
    "ChatResponse",
    # query
    "QueryRequest",
    "QueryResponse",
    "QueryResult",
    # debug
    "HealthResponse",
    "ModelInfo",
    "SessionStateSchema",
    "Message",
    # frontend submission
    "InfoForm",
    "Ticket"
]