# workflows/router.py
import inspect
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Dict, Any, Set

"""
Workflow Router Module.

Provides a lightweight mechanism to register and run asynchronous workflows
by name. Each workflow is stored with its accepted parameters and any fixed
keyword arguments, ensuring that only relevant arguments are passed at runtime.

Main components:
    - WorkflowSpec: Encapsulates a workflow function, fixed arguments, and
      cached accepted parameter names.
    - WorkflowRouter: Registry and dispatcher for workflows, allowing dynamic
      execution by string key.
"""

@dataclass
class WorkflowSpec:
    """
    Specification for a workflow function.

    Stores:
      - The workflow coroutine to call.
      - Any fixed keyword arguments that should always be passed.
      - A cached set of accepted parameter names for efficient filtering.

    Attributes:
        func (Callable[..., Awaitable[None]]): The async function implementing the workflow.
        fixed_kwargs (Dict[str, Any]): Keyword arguments fixed at registration time.
        accepted_params (Set[str]): Cached set of parameter names the workflow function accepts.
    """
    func: Callable[..., Awaitable[None]]
    fixed_kwargs: Dict[str, Any]
    accepted_params: Set[str] = field(default_factory=set)  # ← cache

@dataclass
class WorkflowRouter:
    """
    Router for registering and running workflows by name.

    Allows:
      - Registering async workflows under a string key.
      - Supplying fixed arguments at registration time (e.g., system prompts).
      - Running workflows later with runtime arguments, where only the
        parameters accepted by the workflow function are passed. This means
        you can pass all possible services in, and only the required services will be used.

    Attributes:
        routes (Dict[str, WorkflowSpec]): Mapping of workflow names to their specifications.
    """
    routes: Dict[str, WorkflowSpec] = field(default_factory=dict)

    def register(self, name: str, func: Callable[..., Awaitable[None]], **fixed_kwargs):
        """
        Register a new workflow under a given name.

        Args:
            name (str): Unique identifier for the workflow.
            func (Callable[..., Awaitable[None]]): The async workflow function.
            **fixed_kwargs: Fixed keyword arguments to always pass when running this workflow.
        """
        sig = inspect.signature(func)
        params = set(sig.parameters.keys())
        self.routes[name] = WorkflowSpec(
            func=func,
            fixed_kwargs=fixed_kwargs,
            accepted_params=params
        )

    async def run(self, name: str, **runtime_kwargs):
        """
        Run a registered workflow by name.

        Args:
            name (str): The workflow name.
            **runtime_kwargs: Runtime arguments to pass into the workflow.

        Returns:
            Any: The return value of the workflow function.

        Raises:
            ValueError: If the workflow name is not registered.
        """
        if name not in self.routes:
            raise ValueError(f"Unknown workflow '{name}'")

        spec = self.routes[name]
        merged = {**spec.fixed_kwargs, **runtime_kwargs}

        # Filter by cached accepted parameters
        filtered = {k: v for k, v in merged.items() if k in spec.accepted_params}

        return await spec.func(**filtered)