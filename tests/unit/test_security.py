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
from app.app_utils.security import sterilize_pii, SecurityGuardrailsPlugin


class MockTool:

    def __init__(self, name: str):
        self.name = name


class MockToolActions:

    def __init__(self):
        self.escalate = False


class MockToolContext:

    def __init__(self, state=None):
        self.state = state if state is not None else {}
        self.actions = MockToolActions()


def test_sterilize_pii():
    # Test string sterilization
    assert (
        sterilize_pii("Contact me at owner@downtowncafe.com or call 650-253-0000")
        == "Contact me at [[BUSINESS_OWNER_EMAIL]] or call [[BUSINESS_PHONE_NUMBER]]"
    )

    # Test dictionary sterilization
    data = {
        "email": "owner@downtowncafe.com",
        "phone": "650-253-0000",
        "safe_field": "This is safe.",
    }
    sanitized = sterilize_pii(data)
    assert sanitized["email"] == "[[BUSINESS_OWNER_EMAIL]]"
    assert sanitized["phone"] == "[[BUSINESS_PHONE_NUMBER]]"
    assert sanitized["safe_field"] == "This is safe."

    # Test list sterilization
    items = ["owner@downtowncafe.com", "650-253-0000", "safe"]
    sanitized_items = sterilize_pii(items)
    assert sanitized_items[0] == "[[BUSINESS_OWNER_EMAIL]]"
    assert sanitized_items[1] == "[[BUSINESS_PHONE_NUMBER]]"
    assert sanitized_items[2] == "safe"


@pytest.mark.asyncio
async def test_structural_gating_role():
    plugin = SecurityGuardrailsPlugin(policies_path="policies.yaml")
    tool = MockTool(name="delete_report")

    # User is regular -> blocked
    context_regular = MockToolContext(state={"user_role": "regular"})
    res = await plugin.before_tool_callback(
        tool=tool, tool_args={}, tool_context=context_regular
    )
    assert res is not None
    assert res["status"] == "Policy Violation"

    # User is admin -> allowed
    context_admin = MockToolContext(state={"user_role": "admin"})
    res = await plugin.before_tool_callback(
        tool=tool, tool_args={}, tool_context=context_admin
    )
    assert res is None


@pytest.mark.asyncio
async def test_semantic_gating_pii_interception():
    plugin = SecurityGuardrailsPlugin(policies_path="policies.yaml")
    tool = MockTool(name="write_report")
    context = MockToolContext()

    # Tool call with safe content -> allowed
    safe_args = {"content": "This is a safe report with no PII."}
    res = await plugin.before_tool_callback(
        tool=tool, tool_args=safe_args, tool_context=context
    )
    assert res is None

    # Tool call with raw email -> blocked
    email_args = {"content": "Owner email: owner@downtowncafe.com"}
    res = await plugin.before_tool_callback(
        tool=tool, tool_args=email_args, tool_context=context
    )
    assert res is not None
    assert res["status"] == "Policy Violation"
    assert "Policy Violation" in res["error"]

    # Tool call with raw phone -> blocked
    phone_args = {"content": "Phone: 650-253-0000"}
    res = await plugin.before_tool_callback(
        tool=tool, tool_args=phone_args, tool_context=context
    )
    assert res is not None
    assert res["status"] == "Policy Violation"


@pytest.mark.asyncio
async def test_context_hygiene_after_tool():
    plugin = SecurityGuardrailsPlugin(policies_path="policies.yaml")
    tool = MockTool(name="fetch_actual_attributes")
    context = MockToolContext()

    raw_result = {
        "attributes": ["dineIn"],
        "owner_contact": "owner@downtowncafe.com",
        "phone": "650-253-0000",
    }

    sanitized = await plugin.after_tool_callback(
        tool=tool, tool_args={}, tool_context=context, result=raw_result
    )
    assert sanitized is not None
    assert sanitized["owner_contact"] == "[[BUSINESS_OWNER_EMAIL]]"
    assert sanitized["phone"] == "[[BUSINESS_PHONE_NUMBER]]"
