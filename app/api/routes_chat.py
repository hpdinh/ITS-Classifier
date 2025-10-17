# api/routes_chat.py
"""
Chat Routes API.

Defines FastAPI endpoints for managing frontend chat interactions:
  - `/chat/`   → Send a user message, get assistant reply.
  - `/chat/reset` → Reset the chat session.

These routes depend on:
  - SessionState: For managing chat history and predictions.
  - LLMClient: For generating assistant responses.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from core.dependencies import get_session, get_llm_client
from services import SessionState, LLMClient

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/")
async def chat(
    request: Request,
    session: SessionState = Depends(get_session),
    llm_client: LLMClient = Depends(get_llm_client),
):
    """
    Handle a chat interaction with the assistant.

    Workflow:
        1. Parse the incoming user message.
        2. Ensure the session is initialized (either via predictions or history).
        3. If the message is `"INIT"`, mark the session as initialized (no reply). This takes place when a workflow is initially complete.
        4. Otherwise, append the user message to history.
        5. Call the LLM client with session history for a reply.
        6. Append the assistant reply to session history.
        7. Return the assistant reply in a JSON response.

    Args:
        request (Request): FastAPI request object containing JSON payload with "message".
        session (SessionState): Current chat session (dependency-injected).
        llm_client (LLMClient): LLM client for generating responses (dependency-injected).

    Returns:
        JSONResponse: JSON object with assistant reply under `"reply"` key.

    Raises:
        HTTPException: 400 if message is empty or session not initialized.
        HTTPException: 502 if LLM backend call fails.
    """
    data = await request.json()
    user_msg: str = data.get("message", "").strip()
    
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message")

    if not session.history and not session.predictions:
        raise HTTPException(status_code=400, detail="Session not initialized")
    if user_msg == "INIT":
        if session.init_done:
            return {"reply": None}   # no-op on reloads - shouldn't be called
        session.init_done = True
    else:
        session.add_message("user", user_msg)

    try:
        answer = await llm_client.chat(session.get_openai_history())
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(status_code=502, detail="LLM backend error")

    session.add_message("assistant", answer)

    resp = JSONResponse({"reply": answer})
    return resp

@router.post("/reset")
async def reset_chat(session: SessionState = Depends(get_session)):
    SessionState.delete(session.session_id)
    return {"status": "ok", "message": "Chat reset"}