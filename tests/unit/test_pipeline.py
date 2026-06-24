# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import pipeline


@pytest.mark.asyncio
async def test_sequential_pipeline_execution():
    """Runs the full sequential sub-agent pipeline and verifies state flow and outputs."""
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="app", user_id="test_user", session_id="s_test"
    )

    # Initialize the runner with the pipeline SequentialAgent
    runner = Runner(agent=pipeline, app_name="app", session_service=session_service)

    # Run the pipeline with input "Downtown Cafe"
    messages = []
    async for event in runner.run_async(
        user_id="test_user",
        session_id="s_test",
        new_message=types.Content(
            role="user",
            parts=[types.Part.from_text(text="Perform gap analysis for Downtown Cafe")],
        ),
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                messages.append(event.content.parts[0].text)

    # Load session state to verify data propagated between sub-agents
    loaded_session = await session_service.get_session(
        app_name="app", user_id="test_user", session_id="s_test"
    )
    state = loaded_session.state

    # 1. Verify Search Agent details
    assert state.get("place_id") == "mock_downtown_cafe"
    assert state.get("primary_category") == "Coffee Shop"

    # 2. Verify Data Agent details (fetched via MCP and saved)
    assert "actual_attributes" in state
    assert "possible_attributes" in state
    assert len(state["actual_attributes"]) == 3
    assert len(state["possible_attributes"]) == 15

    # 3. Verify Analyst Agent generated a report and called write_report
    assert len(messages) > 0
    events_str = str([e.model_dump() for e in loaded_session.events]).lower()
    assert "write_report" in events_str
    assert "attribute_gap_report" in events_str
