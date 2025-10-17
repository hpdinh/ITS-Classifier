from .router import WorkflowRouter
from .escalation_workflow import run_escalation
from .troubleshooting_workflow import run_troubleshooting

from core.config import PromptSettings
prompts = PromptSettings()

workflow_router = WorkflowRouter()

workflow_router.register("escalation", run_escalation, SYSTEM_PROMPT=prompts.get("triage"))
workflow_router.register("troubleshooting", run_troubleshooting, SYSTEM_PROMPT=prompts.get("troubleshooting"))