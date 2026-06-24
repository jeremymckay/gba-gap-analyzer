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

import os
import re
import logging
from typing import Any, Optional, Dict, List
import yaml

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import BaseTool, ToolContext
from google.genai import Client

logger = logging.getLogger("gba_gap_analyzer_security")

# Context Hygiene PII patterns
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?[2-9]\d{2}[-.\s]?\d{4}\b"
)


def sterilize_pii(val: Any) -> Any:
    """Recursively replaces raw emails and phone numbers with bracketed placeholders."""
    if isinstance(val, str):
        val = EMAIL_REGEX.sub("[[BUSINESS_OWNER_EMAIL]]", val)
        val = PHONE_REGEX.sub("[[BUSINESS_PHONE_NUMBER]]", val)
        return val
    elif isinstance(val, dict):
        return {k: sterilize_pii(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [sterilize_pii(v) for v in val]
    return val


class SecurityGuardrailsPlugin(BasePlugin):
    """ADK Plugin implementing Context Hygiene, Structural Gating, and Semantic Gating."""

    def __init__(self, policies_path: str = "policies.yaml"):
        super().__init__(name="security_guardrails")
        self.policies_path = policies_path
        self.policies = self._load_policies()

    def _load_policies(self) -> Dict[str, Any]:
        """Loads structural gating policies from policies.yaml."""
        if os.path.exists(self.policies_path):
            try:
                with open(self.policies_path, "r") as f:
                    config = yaml.safe_load(f)
                    return config.get("policies", {})
            except Exception as e:
                logger.error(f"Error loading policies.yaml: {str(e)}")
        else:
            logger.warning(
                f"policies.yaml not found at {self.policies_path}. Using default policies."
            )
        return {}

    def get_tool_policy(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Retrieves policy definition for a specific tool."""
        restricted_tools = self.policies.get("restricted_tools", [])
        for tool_policy in restricted_tools:
            if tool_policy.get("name") == tool_name:
                return tool_policy
        return None

    async def run_semantic_referee(self, tool_name: str, tool_args: dict) -> bool:
        """Invokes the LLM referee (Semantic Gating) to check if the action violates privacy.

        Returns True if a privacy violation is detected.
        """
        # Determine if semantic referee is globally enabled
        global_settings = self.policies.get("global_settings", {})
        if not global_settings.get("semantic_referee_enabled", True):
            return False

        try:
            client = Client()
            prompt = f"""
You are a Security Policy Referee. Your task is to analyze a proposed tool call and determine if its arguments disclose private personally identifiable information (PII) such as specific raw email addresses, private phone numbers, or personal contact details.

Tool Name: {tool_name}
Tool Arguments: {tool_args}

Analyze the content. Does it contain sensitive raw PII (like a raw email address 'user@domain.com' or phone number '123-456-7890')? Note: bracketed placeholders like '[[BUSINESS_OWNER_EMAIL]]' are safe and NOT violations.

Respond in JSON format (do not include markdown formatting code block, just raw JSON):
{{
  "violation": true/false,
  "reason": "explanation of decision"
}}
"""
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )

            import json

            result = json.loads(response.text)
            logger.info(f"Semantic referee result: {result}")
            return result.get("violation", False)
        except Exception as e:
            logger.error(f"Error calling Semantic Referee LLM: {str(e)}")
            # Fallback to local regex-based check if LLM referee fails or credentials are not set
            content_str = str(tool_args)
            if EMAIL_REGEX.search(content_str) or PHONE_REGEX.search(content_str):
                logger.info(
                    "Local fallback: regex matched raw email or phone. Flagging violation."
                )
                return True
            return False

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict]:
        """Intercepts tool calls before execution (Structural & Semantic Gating)."""
        tool_name = tool.name
        policy = self.get_tool_policy(tool_name)

        if not policy:
            return None

        # 1. Structural Gate: Role checking
        allowed_roles = policy.get("allowed_roles")
        if allowed_roles:
            user_role = tool_context.state.get("user_role", "regular")
            if user_role not in allowed_roles:
                logger.warning(
                    f"Structural Gate: Role '{user_role}' blocked from executing '{tool_name}'"
                )
                return {
                    "status": "Policy Violation",
                    "error": f"Unauthorized: Role '{user_role}' is not allowed to call '{tool_name}'.",
                }

        # 2. Structural/Semantic Gate: PII writing check
        block_pii = policy.get("block_pii", False)
        if block_pii:
            content_str = str(tool_args)
            if EMAIL_REGEX.search(content_str) or PHONE_REGEX.search(content_str):
                # Call Semantic Referee to confirm violation
                is_violation = await self.run_semantic_referee(tool_name, tool_args)
                if is_violation:
                    logger.warning(
                        f"Semantic Gate: Intercepted tool '{tool_name}' containing PII."
                    )
                    return {
                        "status": "Policy Violation",
                        "error": "Policy Violation: Writing private phone numbers or owner emails to a public report is strictly prohibited.",
                    }

        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> Optional[dict]:
        """Intercepts tool output before returning to agent context (Context Hygiene)."""
        logger.info(f"Context Hygiene: Sanitizing outputs for tool '{tool.name}'")
        return sterilize_pii(result)
