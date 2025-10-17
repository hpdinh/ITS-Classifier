# workflows/escalation_workflow.py

"""
Escalation Workflow Module.

Defines the workflow for handling escalation requests. This workflow:
  - Uses an LLM client to rephrase input if necessary.
  - Classifies the issue using a classifier (local or API-based).
  - Looks up related assignment groups from Confluence.
  - Updates the session state with predictions, knowledge base snippets,
    and context messages for downstream use.
"""


from services import SessionState, ClassifierService, SnowAssignmentGroupsRetriever, LLMClient
from schemas import InfoForm
import logging


logger = logging.getLogger(__name__)

async def run_escalation(
    session: SessionState,
    form: InfoForm,
    classifier: ClassifierService,
    prediction_retriever: SnowAssignmentGroupsRetriever,
    llm_client: LLMClient,
    SYSTEM_PROMPT: str,
):
    """
    Run the escalation workflow.

    Steps:
        1. Convert the form into classification text.
        2. Optionally rephrase the text if the input type is a call.
        3. Predict assignment group(s) using the classifier.
        4. Retrieve matching assignment group information from Confluence.
        5. Reset and update the session with system prompt, user input, predictions,
           and retrieved knowledge base snippets.

    Args:
        session (SessionState): The current session state to update.
        form (InfoForm): Input form containing ticket/call details.
        classifier (ClassifierService): Classifier service for prediction.
        prediction_retriever (SnowAssignmentGroupsRetriever): Retriever for assignment group info.
        llm_client (LLMClient): LLM client for rephrasing input text.
        SYSTEM_PROMPT (str): The system prompt used to initialize assistant context.

    Effects:
        Updates the `session` with predictions, KB snippets, and context messages.
    """

    prediction_text = form.to_classification_text()

    if form.input_type=='call':
        prediction_text = await llm_client.rephrase(prediction_text, mode='classification')
    prediction = await classifier.predict(prediction_text)

    kb_snippets = prediction_retriever.prediction_lookup(prediction) if prediction else []

    session.reset()
    session.add_message("system", SYSTEM_PROMPT)
    session.add_message("user", f"Please triage this ticket or issue:\n{form.to_full_info_text()}", False)
    session.set_predictions(prediction)

    context_msgs = []
    if prediction:
        context_msgs.append("Top Predictions: " + ", ".join(prediction))
    if kb_snippets:
        context_msgs.append("KB Snippets:\n- " + "\n- ".join(kb_snippets))
    session.add_message("assistant", "\n\n".join(context_msgs) or "No KB snippets available.", False)

    # Store structured state
    session.predictions = prediction
    session.kb_snippets = kb_snippets
