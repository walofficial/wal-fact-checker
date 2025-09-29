"""Custom A2A AgentExecutor for WAL Fact Checker.

Implements the Agent2Agent (A2A) Protocol's AgentExecutor interface and
bridges requests to the Google ADK `Runner` hosting our orchestrator agent.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    Artifact,
    Message,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from google.adk.a2a.converters.event_converter import convert_event_to_a2a_events
from google.adk.a2a.converters.part_converter import (
    convert_a2a_part_to_genai_part,
    convert_genai_part_to_a2a_part,
)
from google.adk.a2a.converters.request_converter import (
    convert_a2a_request_to_adk_run_args,
)
from google.adk.a2a.converters.utils import _get_adk_metadata_key
from google.adk.runners import Runner
from pydantic import BaseModel
from typing_extensions import override

logger = logging.getLogger(__name__)


class WalAgentExecutorConfig(BaseModel):
    """Configuration for `WalAgentExecutor`.

    Attributes:
        a2a_part_converter: Converter for A2A part → GenAI part
        gen_ai_part_converter: Converter for GenAI part → A2A part
    """

    a2a_part_converter: Any = convert_a2a_part_to_genai_part
    gen_ai_part_converter: Any = convert_genai_part_to_a2a_part


class WalAgentExecutor(AgentExecutor):
    """Custom AgentExecutor that runs our ADK Runner for A2A requests."""

    def __init__(
        self,
        *,
        runner: Runner | Callable[..., Runner | Awaitable[Runner]],
        config: WalAgentExecutorConfig | None = None,
    ) -> None:
        super().__init__()
        self._runner: Runner | Callable[..., Runner | Awaitable[Runner]] = runner
        self._config: WalAgentExecutorConfig = config or WalAgentExecutorConfig()

    async def _resolve_runner(self) -> Runner:
        """Resolve a `Runner` from a possibly callable provider."""
        if isinstance(self._runner, Runner):
            return self._runner
        if callable(self._runner):
            result = self._runner()
            if inspect.iscoroutine(result):
                resolved = await result
            else:
                resolved = result
            self._runner = resolved
            return resolved
        raise TypeError(f"Unsupported runner type: {type(self._runner)}")

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel handling is not supported for now."""
        raise NotImplementedError("Cancellation is not supported")

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the request and stream A2A events to the queue."""
        if not context.message:
            raise ValueError("A2A request must include a message")

        # If this is a new task, publish submitted
        if not context.current_task:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    status=TaskStatus(
                        state=TaskState.submitted,
                        message=context.message,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                    context_id=context.context_id,
                    final=False,
                )
            )

        try:
            await self._handle_request(context, event_queue)
        except Exception as err:
            logger.exception("A2A request handling failed")
            await self._publish_failure(context, event_queue, err)

    async def _handle_request(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        runner = await self._resolve_runner()

        run_args = convert_a2a_request_to_adk_run_args(
            context, self._config.a2a_part_converter
        )

        session = await self._prepare_session(context, run_args, runner)
        invocation_ctx = runner._new_invocation_context(  # pylint: disable=protected-access
            session=session,
            new_message=run_args["new_message"],
            run_config=run_args["run_config"],
        )

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                status=TaskStatus(
                    state=TaskState.working,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                context_id=context.context_id,
                final=False,
                metadata={
                    _get_adk_metadata_key("app_name"): runner.app_name,
                    _get_adk_metadata_key("user_id"): run_args["user_id"],
                    _get_adk_metadata_key("session_id"): run_args["session_id"],
                },
            )
        )

        # Stream ADK events and convert to A2A events
        last_status_message: Message | None = None
        async for adk_event in runner.run_async(**run_args):
            for a2a_event in convert_event_to_a2a_events(
                adk_event,
                invocation_ctx,
                context.task_id,
                context.context_id,
                self._config.gen_ai_part_converter,
            ):
                # Track last task status message if present
                if (
                    isinstance(a2a_event, TaskStatusUpdateEvent)
                    and a2a_event.status.message is not None
                ):
                    last_status_message = a2a_event.status.message
                # await event_queue.enqueue_event(a2a_event)

        # Publish finalization
        await self._publish_completion(context, event_queue, last_status_message)

    async def _prepare_session(
        self, context: RequestContext, run_args: dict[str, Any], runner: Runner
    ) -> Any:
        session_id = run_args["session_id"]
        user_id = run_args["user_id"]
        session = await runner.session_service.get_session(
            app_name=runner.app_name, user_id=user_id, session_id=session_id
        )
        if session is None:
            session = await runner.session_service.create_session(
                app_name=runner.app_name,
                user_id=user_id,
                state={},
                session_id=session_id,
            )
            run_args["session_id"] = session.id
        return session

    async def _publish_completion(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        last_status_message: Message | None,
    ) -> None:
        if last_status_message and last_status_message.parts:
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    task_id=context.task_id,
                    last_chunk=True,
                    context_id=context.context_id,
                    artifact=Artifact(
                        artifact_id=str(uuid.uuid4()),
                        parts=last_status_message.parts,
                    ),
                )
            )
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    status=TaskStatus(
                        state=TaskState.completed,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                    context_id=context.context_id,
                    final=True,
                )
            )
        else:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    status=TaskStatus(
                        state=TaskState.completed,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        message=Message(
                            message_id=str(uuid.uuid4()),
                            role=Role.agent,
                            parts=[TextPart(text="Done")],
                        ),
                    ),
                    context_id=context.context_id,
                    final=True,
                )
            )

    async def _publish_failure(
        self, context: RequestContext, event_queue: EventQueue, err: Exception
    ) -> None:
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                status=TaskStatus(
                    state=TaskState.failed,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    message=Message(
                        message_id=str(uuid.uuid4()),
                        role=Role.agent,
                        parts=[TextPart(text=str(err))],
                    ),
                ),
                context_id=context.context_id,
                final=True,
            )
        )
