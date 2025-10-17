# services/core_utils.py
from fastapi import Response
from services.session_service import SessionState
from typing import List, Optional

def validate_form(short_description: Optional[str], description: Optional[str]) -> List[str]:
    """
    Validate the required form fields for ticket submission.

    Args:
        short_description (Optional[str]): The short description provided by the user.
        description (Optional[str]): The detailed description provided by the user.

    Returns:
        List[str]: A list of error messages. Returns an empty list if all fields are valid.
    """
    errors = []
    if not short_description or not short_description.strip():
        errors.append("Short description is required.")
    if not description or not description.strip():
        errors.append("Description is required.")
    return errors

def attach_session_cookie(response: Response, session: SessionState) -> Response:
    """
    Attach a session cookie to the HTTP response and persist the session state.

    Args:
        response (Response): The FastAPI Response object to which the cookie is added.
        session (SessionState): The session state to save and associate with the cookie.

    Returns:
        Response: The updated response with the session cookie set.
    """
    response.set_cookie(
        key="session_id",
        value=session.session_id,
        httponly=True,
        max_age=3600,
        samesite="lax",
    )
    session.save()
    return response
