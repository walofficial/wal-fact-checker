# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2AClient
from a2a.client.card_resolver import A2ACardResolver
from a2a.types import (
    DataPart,
    MessageSendConfiguration,
    MessageSendParams,
    SendMessageRequest,
    SendMessageSuccessResponse,
    Task,
)
from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="realitycheckagent",
    description="API for interacting with the Agent realitycheckagent",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BODY_TEST_SEND = Body(
    default=None,
    description=(
        "Optional JSON body: { 'text': 'your message' }. Defaults to a USD→INR query."
    ),
)


@app.post("/test/send")
async def test_send_message(
    request: Request, body: dict[str, Any] | None = BODY_TEST_SEND
) -> dict[str, Any]:
    """Send a test message via the A2A API and return the response JSON."""

    user_text: str = (
        body.get("text")
        if body and isinstance(body.get("text"), str)
        else "how much is 10 USD in INR?"
    )

    async with httpx.AsyncClient(timeout=600.0) as httpx_client:
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url="http://localhost:8003",
        )
        agent_card = await resolver.get_agent_card()
        client = A2AClient(
            httpx_client=httpx_client,
            agent_card=agent_card,
            url="http://localhost:8003",
        )

        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(
                **{
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": user_text}],
                        "messageId": uuid4().hex,
                    },
                    "configuration": MessageSendConfiguration(
                        history_length=0, blocking=True
                    ),
                }
            ),
        )

        response = await client.send_message(request)

        # Extract the last artifact whose last part is a FunctionResponse

        match response.root:
            case SendMessageSuccessResponse() as success_response:
                match success_response.result:
                    case Task() as task:
                        artifacts = task.artifacts or []
                        for artifact in reversed(artifacts):
                            parts = artifact.parts or []
                            for part in reversed(parts):
                                match part.root:
                                    case DataPart() as data_part:
                                        return data_part.data.get("response")

        return response.root.model_dump()
