"""Adapter to wrap litellm.acompletion for its-hub algorithms."""

from __future__ import annotations

import asyncio
from typing import Any

try:
    from its_hub.base import AbstractLanguageModel
except ImportError:
    raise ImportError(
        "its-hub is required for ITS algorithms. Install with: pip install its-hub"
    )

import litellm


class LiteLLMLanguageModel(AbstractLanguageModel):
    """
    Wrapper around litellm.acompletion for its-hub algorithms.

    This adapter allows its-hub algorithms (Self-Consistency, Best-of-N, etc.)
    to use litellm as their language model backend, enabling inference-time
    scaling across all 100+ LiteLLM-supported providers.
    """

    def __init__(self, model: str, **litellm_kwargs: Any):
        """
        Initialize the LiteLLM language model wrapper.

        Args:
            model: LiteLLM model name (e.g., "gpt-4", "anthropic/claude-3-5-sonnet")
            **litellm_kwargs: Additional parameters for litellm.acompletion
                             (temperature, api_key, max_tokens, etc.)
        """
        self.model = model
        self.litellm_kwargs = litellm_kwargs

    def _convert_to_dict(self, message: Any) -> dict:
        """Convert ChatMessage or dict to dict format."""
        if isinstance(message, dict):
            return message
        elif hasattr(message, "to_dict"):
            # ChatMessage object from its-hub
            return message.to_dict()
        else:
            # Fallback - try to extract attributes
            return {
                "role": message.role if hasattr(message, "role") else "user",
                "content": message.content if hasattr(message, "content") else "",
            }

    async def agenerate(
        self,
        messages: list[dict] | list[list[dict]],
        stop: str | None = None,
        **kwargs: Any,
    ) -> dict | list[dict]:
        """
        Generate response(s) asynchronously using litellm.acompletion.

        Args:
            messages: Single conversation (list[dict]) or batch of conversations (list[list[dict]])
                     Each message dict format: {"role": "user/assistant", "content": "..."}
            stop: Optional stop sequence for generation
            **kwargs: Additional generation params (override litellm_kwargs)

        Returns:
            Single response dict or list of response dicts (for batched input)
            Response dict format: {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        # Merge kwargs with priority: call-time kwargs > init-time litellm_kwargs
        call_kwargs = {**self.litellm_kwargs, **kwargs}
        if stop is not None:
            call_kwargs["stop"] = stop

        # Detect batch vs single
        is_batch = isinstance(messages[0], list) if messages else False

        # Convert ChatMessage objects to dicts if needed
        if is_batch:
            messages = [[self._convert_to_dict(msg) for msg in conv] for conv in messages]
        else:
            messages = [self._convert_to_dict(msg) for msg in messages]

        if is_batch:
            # Batch call - make N concurrent calls with asyncio.gather
            tasks = [
                litellm.acompletion(model=self.model, messages=msg, **call_kwargs)
                for msg in messages
            ]
            responses = await asyncio.gather(*tasks)

            # Convert LiteLLM ModelResponse objects to its-hub format
            return [self._convert_response(r) for r in responses]
        else:
            # Single call
            response = await litellm.acompletion(
                model=self.model, messages=messages, **call_kwargs
            )
            return self._convert_response(response)

    def _convert_response(self, litellm_response: Any) -> dict:
        """
        Convert LiteLLM ModelResponse to its-hub response format.

        Args:
            litellm_response: LiteLLM ModelResponse object

        Returns:
            dict with {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        message = litellm_response.choices[0].message

        # Build response dict
        response_dict = {
            "role": message.role,
            "content": message.content,
        }

        # Include tool_calls if present
        if hasattr(message, "tool_calls") and message.tool_calls is not None:
            # Convert tool calls to dict format
            tool_calls_list = []
            for tc in message.tool_calls:
                if hasattr(tc, "model_dump"):
                    tool_calls_list.append(tc.model_dump())
                elif isinstance(tc, dict):
                    tool_calls_list.append(tc)
                else:
                    # Fallback: convert to dict manually
                    tool_calls_list.append(
                        {
                            "id": tc.id if hasattr(tc, "id") else None,
                            "type": tc.type if hasattr(tc, "type") else "function",
                            "function": {
                                "name": tc.function.name
                                if hasattr(tc.function, "name")
                                else None,
                                "arguments": tc.function.arguments
                                if hasattr(tc.function, "arguments")
                                else None,
                            },
                        }
                    )
            response_dict["tool_calls"] = tool_calls_list

        return response_dict
