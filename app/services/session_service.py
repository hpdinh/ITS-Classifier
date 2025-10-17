from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any, ClassVar
import json
from core.config import SessionSettings
from schemas import Ticket

@dataclass
class SessionState:
    """
    Represents the session state for a user, including chat history,
    predictions, form data, related tickets, and error messages.
    
    The state can be persisted either in Redis or in-memory storage,
    depending on configuration.
    """
    session_id: str

    # Chat history
    history: List[Dict[str, str]] = field(default_factory=list)

    # Classifier predictions and the snippets from relevant assignment groups
    predictions: List[str] = field(default_factory=list)
    kb_snippets: List[str] = field(default_factory=list)


    # UI state
    errors: Optional[List[str]] = None
    show_chat: bool = False

    # Stores all the settings on the form that are being used
    form_data: Dict[str, Any] = field(default_factory=dict)

    # Helper for running the first chat message
    init_done: bool = False

    # Relevant tickets, stored in json from the Ticket schema
    related_tickets: List[Dict[str, Any]] = field(default_factory=list)

    _settings: ClassVar = SessionSettings()
    _memory_store: ClassVar[Dict[str, str]] = {}

    # Sets up Redis (should be used in production)
    if _settings.use_redis:
        import redis
        _redis_client: ClassVar = redis.Redis(
            host=_settings.redis_host,
            port=_settings.redis_port,
            db=_settings.redis_db,
            decode_responses=True,
        )

    def reset(self, new_ticket: bool=True):
        """
        Reset the session state to its initial values.

        Args:
            new_ticket (bool, optional): If True, clears form data in addition 
                to chat history, predictions, snippets, and related tickets. 
                Defaults to True.
        """
        self.history.clear()
        self.kb_snippets.clear()
        self.predictions.clear()
        self.related_tickets.clear()

        if new_ticket:
            self.form_data.clear()

        self.errors = None
        self.show_chat = False
        self.init_done = False
        self.save()

    def add_message(self, role: str, content: str, show_msg: str = True):
        """
        Add a message to the session history.

        Args:
            role (str): The role of the speaker (e.g., "user", "assistant").
            content (str): The message content.
            show_msg (bool, optional): Whether to display this message in the UI.
                Defaults to True.
        """
        self.history.append({"role": role, "content": content, "show_msg": show_msg})
        self.save()

    def set_predictions(self, preds: List[str]):
        """
        Set model predictions for the current session.

        Args:
            preds (List[str]): A list of prediction labels or strings.
        """
        self.predictions = preds
        self.save()

    def get_openai_history(self):
        """
        Convert chat history into a format compatible with OpenAI API calls (standard format for all LLM APIs).

        Returns:
            List[Dict[str, str]]: List of messages containing "role" and "content".
        """

        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.history if "role" in m and "content" in m
        ]

    def to_dict(self):
        """
        Convert the session state into a serializable dictionary.

        Returns:
            Dict[str, Any]: The session state as a dictionary, excluding session_id.
        """
        d = asdict(self)
        d.pop("session_id", None)  # don’t serialize this
        return d
    
    def pop_errors(self) -> Optional[List[str]]:
        """
        Retrieve and clear error messages for this session.

        Returns:
            Optional[List[str]]: List of error strings if any exist, otherwise None.
        """
        errs = self.errors
        self.errors = None
        self.save()
        return errs
    
    def add_tickets(self, tickets: List[Ticket]):
        """
        Add tickets to the session state.

        Args:
            tickets (List[Ticket]): List of Ticket objects to be added.
        """
        for t in tickets:
            self.related_tickets.append(t.model_dump())
        self.save()

    def get_rag_context(self) -> str:
        """
        Build a Retrieval-Augmented Generation (RAG) context string from related tickets.

        Returns:
            str: A string with ticket snippets joined by separators, or an empty string if no tickets.
        """
        """Combine related_tickets into a RAG-friendly context string."""
        if not self.related_tickets:
            return ""
        return "\n\n---\n\n".join(
            Ticket(**t).to_rag_snippet()
            for t in self.related_tickets
        )
    
    def get_related_ticket_links(self) -> list[str]:
        """
        Build a list of Markdown links to related tickets.

        Returns:
            List[str]: List of ticket links in Markdown format.
        """

        base_url = "https://support.ucsd.edu/nav_to.do?uri=task.do?sysparm_query=number="
        links = []

        for t in self.related_tickets:

            number = t.get("number")            
            if number:
                # Build Markdown link with optional desc
                link = f"[{number}]({base_url}{number})"
                links.append(link)
            else:
                # fallback to avoid "undefined"
                links.append("(unknown ticket)")

        return links
    
    @classmethod
    def from_dict(cls, session_id: str, data: Dict) -> "SessionState":
        """
        Reconstruct a SessionState object from a dictionary.

        Args:
            session_id (str): The session identifier.
            data (Dict): Dictionary containing serialized session state.

        Returns:
            SessionState: Reconstructed session state.
        """
        return cls(session_id=session_id, **data)

    def save(self):
        """
        Persist the current session state to Redis or in-memory storage.
        """
        data = json.dumps(self.to_dict())
        if self._settings.use_redis:
            self._redis_client.setex(
                self._settings.session_prefix + self.session_id,
                self._settings.session_ttl,
                data,
            )
        else:
            self._memory_store[self.session_id] = data

    @classmethod
    def load(cls, session_id: str) -> "SessionState":
        """
        Load a session state from Redis or in-memory storage.

        Args:
            session_id (str): The session identifier.

        Returns:
            SessionState: Loaded session state, or a new empty state if none exists.
        """
        if cls._settings.use_redis:
            raw = cls._redis_client.get(cls._settings.session_prefix + session_id)
        else:
            raw = cls._memory_store.get(session_id)
        if raw:
            return cls.from_dict(session_id, json.loads(raw))
        return cls(session_id)

    @classmethod
    def delete(cls, session_id: str):
        """
        Delete a session state from Redis or in-memory storage.

        Args:
            session_id (str): The session identifier.
        """
        if cls._settings.use_redis:
            cls._redis_client.delete(cls._settings.session_prefix + session_id)
        else:
            cls._memory_store.pop(session_id, None)
