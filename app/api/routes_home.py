# api/routes_home.py
"""
Home Routes API.

Defines FastAPI routes for the home page, form submission, session reset, 
and updating sessions. These routes handle both the initial user workflow 
form and the chat-based troubleshooting/escalation interface.

Endpoints:
    - GET `/`       → Render form or chat view depending on session state.
    - POST `/`      → Handle form submission, run workflow, redirect to GET.
    - GET `/reset`  → Reset the entire session and redirect to form.
    - GET `/update` → Reset session while keeping form data, redirect to form.
"""
import logging
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette import status

from core.dependencies import get_classifier, get_session, get_groups_retriever, get_llm_client, get_ticket_query
from services.service_utils import attach_session_cookie
from services import ClassifierService, SessionState
from services.workflows import WorkflowRouter

from schemas import InfoForm


templates = Jinja2Templates(directory="assets/templates")
logger = logging.getLogger(__name__)

router = APIRouter()

# Register new routes in services/workflows/__init__.py
workflow_router = WorkflowRouter()


# ---------------------------
# GET /
# ---------------------------
@router.get("/", response_class=HTMLResponse)
async def home_page(request: Request, session: SessionState = Depends(get_session)):
    """
    Render either the empty workflow form or the chat view.

    The page content depends on session state:
      - Default: show empty form.
      - If form data exists: prepopulate the form.
      - If predictions/snippets exist: include them in context.
      - If chat has started: render chat state.

    Args:
        request (Request): FastAPI request object.
        session (SessionState): Current session state.

    Returns:
        HTMLResponse: Rendered HTML response for the home page.
    """
    context = {
        "request": request,
        "workflow": "troubleshooting",
        "input_type": "ticket",
        "errors": None,
        "show_chat": False,
        "history": []
    }

    errors = session.pop_errors()
    # If errors exist in session (flash-style), pop and show once
    if errors:
        context["errors"] = errors

    
    if session.form_data:
        context.update(session.form_data)
        context['ticket_summary'] = InfoForm(**session.form_data).to_markdown()
    # Predictions + snippets (only if classifier has run)
    if session.predictions:
        context["prediction"] = session.predictions
    if session.kb_snippets:
        context["kb_snippets"] = session.kb_snippets
    if session.related_tickets:
        context['related_tickets'] = session.get_related_ticket_links()
    # If predictions/chat are in session, render chat view
    if session.show_chat:
        context["show_chat"] = True
        context["history"] = session.history
        context["init_done"] = session.init_done
    return attach_session_cookie(
        templates.TemplateResponse("index.html", context),
        session
    )

# ---------------------------
# POST /
# ---------------------------
@router.post("/", response_class=HTMLResponse)
async def submit_information(
    request: Request,
    form: InfoForm = Depends(InfoForm.as_form),
    classifier: ClassifierService = Depends(get_classifier),
    session: SessionState = Depends(get_session),
    prediction_retriever=Depends(get_groups_retriever),
    snow_retriever=Depends(get_ticket_query),
    llm_client=Depends(get_llm_client)
):
    """
    Handle form submission and run the appropriate workflow.

    Steps:
        1. Validate the submitted form fields.
        2. If invalid, store errors in session and redirect.
        3. Check if the workflow is registered; if not, return an error.
        4. Run the registered workflow with provided dependencies.
        5. Update session to show chat view.
        6. Redirect back to GET `/`.

    Args:
        request (Request): FastAPI request object.
        form (InfoForm): Submitted form data.
        classifier (ClassifierService): Classifier dependency.
        session (SessionState): Session state dependency.
        prediction_retriever: Confluence-based assignment group retriever.
        snow_retriever: Qdrant-based ticket retriever.
        llm_client: LLM client for rephrasing and chat.

    Returns:
        RedirectResponse: Redirect to GET `/` with updated session state.
    """

    # Validate form
    errors = form.validate()
    if errors:
        session.errors = errors
        session.show_chat = False
        session.form_data = form.model_dump()
        resp = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
        return attach_session_cookie(resp, session)

    if form.workflow not in workflow_router.routes:
        session.errors = [f"Workflow not registered: {form.workflow}"]
        session.show_chat = False
        session.form_data = form.model_dump()
        resp = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
        return attach_session_cookie(resp, session)
    
    await workflow_router.run(
        form.workflow, 
        session=session, form=form, classifier=classifier,
        llm_client=llm_client, snow_retriever=snow_retriever, 
        prediction_retriever=prediction_retriever
    )
    # Redirect to GET /
    session.show_chat = True
    session.form_data = form.model_dump()

    resp = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    return attach_session_cookie(resp, session)


@router.get("/reset", response_class=HTMLResponse)
async def reset_session(session: SessionState = Depends(get_session)):
    """
    Reset the session completely.

    Clears session state (including form data) and redirects to home form.

    Args:
        session (SessionState): Session state dependency.

    Returns:
        RedirectResponse: Redirect to GET `/`.
    """

    session.reset()

    # redirect to form page
    return RedirectResponse("/", status_code=303)


@router.get("/update", response_class=HTMLResponse)
async def reset_session(session: SessionState = Depends(get_session)):
    """
    Reset session without clearing form data.

    Args:
        session (SessionState): Session state dependency.

    Returns:
        RedirectResponse: Redirect to GET `/`.
    """
    session.reset(new_ticket=False)

    return RedirectResponse("/", status_code=303)
