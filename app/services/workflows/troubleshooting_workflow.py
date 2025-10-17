"""
Troubleshooting Workflow Module.

Defines the workflow for handling troubleshooting requests. This workflow:
  - Retrieves similar ServiceNow tickets using dense embeddings via Qdrant.
  - Converts retrieved results into structured `Ticket` objects.
  - Builds a RAG-style context string for the LLM.
  - Updates the session state with related tickets and structured LLM messages.
"""

from services import SessionState, SnowTicketQuery
from schemas import InfoForm
import logging

from schemas import Ticket

logger = logging.getLogger(__name__)

async def run_troubleshooting(
    session: SessionState,
    form: InfoForm,
    snow_retriever: SnowTicketQuery,
    SYSTEM_PROMPT: str,
):
    """
    Run the troubleshooting workflow.

    Steps:
        1. Convert the form into a full query text.
        2. Use Qdrant dense search to retrieve the top-k most relevant tickets.
        3. Convert search results into structured `Ticket` objects.
        4. Build a context string by concatenating ticket snippets.
        5. Reset and update the session with tickets, system prompt, and user input.

    Args:
        session (SessionState): The current session state to update.
        form (InfoForm): Input form containing ticket/call details.
        snow_retriever (SnowTicketQuery): Qdrant retriever for related ServiceNow tickets.
        SYSTEM_PROMPT (str): System prompt used to initialize assistant context.

    Effects:
        Updates the `session` with related tickets and LLM-compatible context messages.
    """
    query_text = form.to_full_info_text()
    hits = await snow_retriever.dense_search(query_text=query_text, top_k=3, input_type=form.input_type)
    tickets = [Ticket.from_scored_point(p) for p in hits]

    context = "\n\n---\n\n".join(t.to_rag_snippet() for t in tickets)
    
    session.reset()
    session.add_tickets(tickets=tickets)

    # Build LLM messages
    session.add_message('system', SYSTEM_PROMPT, show_msg=False)
    session.add_message('user',  (
                f"Relevant Tickets:\n{context}\n\n"
                f"User Issue:\n{query_text}"
            ), show_msg=False)
    
