from typing import Optional
from pydantic import BaseModel, Field
from textwrap import shorten

class Ticket(BaseModel):
    """
    Data model for a ServiceNow ticket.

    Represents ticket metadata and text content as retrieved from Qdrant
    or other sources. Provides utilities to construct tickets from scored
    search results and to format them into Retrieval-Augmented Generation (RAG)
    friendly text snippets.

    Attributes:
        id (int): Unique identifier of the ticket (from vector database).
        score (float): Similarity or ranking score of the ticket.
        number (Optional[str]): Human-readable ServiceNow ticket number.
        assignment_group (Optional[str]): Assignment group responsible for the ticket.
        history (Optional[str]): Free-text ticket history or conversation log.
        short_description (Optional[str]): Short description field.
        description (Optional[str]): Full description field.
    """
    id: int
    score: float
    number: Optional[str] = Field(default=None)
    assignment_group: Optional[str] = Field(default=None)
    history: Optional[str] = Field(default=None)
    short_description: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)

    @classmethod
    def from_scored_point(cls, point) -> "Ticket":
        """
        Factory method to create a Ticket from a Qdrant ScoredPoint object.

        Args:
            point: A Qdrant `ScoredPoint` containing the ticket's vector
                match result, including `id`, `score`, and `payload`.

        Returns:
            Ticket: A populated `Ticket` instance with extracted fields.
        """
        combined = point.payload.get("combined", "")
        short_desc, desc = None, None

        if "####" in combined:
            parts = combined.split("####", 1)
            short_desc = parts[0].strip()
            desc = parts[1].strip()
        elif combined:
            # fallback: treat entire thing as short_description
            desc = combined.strip()

        return cls(
            id=point.id,
            score=point.score,
            number=point.payload.get("number"),
            assignment_group=point.payload.get("assignment_group"),
            history=point.payload.get("history"),
            short_description=short_desc,
            description=desc,
        )

    def to_rag_snippet(self, truncate_history: int = 4096) -> str:
        """
        Format this ticket into a Retrieval-Augmented Generation (RAG) snippet.

        Produces a structured, human-readable string representation of the ticket
        that can be appended to LLM context.

        Args:
            truncate_history (int, optional): Maximum length of the history string.
                If the history exceeds this length, it will be truncated with a
                placeholder. Defaults to 4096 characters.

        Returns:
            str: A formatted string containing ticket number, short description,
            description, and (possibly truncated) history.
        """
        hist = self.history.strip() if self.history else "N/A"
        if truncate_history and len(hist) > truncate_history:
            hist = shorten(hist, width=truncate_history, placeholder=" ...[truncated]")

        return (
            f"Ticket Number: {self.number or 'N/A'}\n"
            f"Short: {self.short_description or 'N/A'}\n"
            f"Desc: {self.description or 'N/A'}\n"
            f"History:\n {hist}"
        )