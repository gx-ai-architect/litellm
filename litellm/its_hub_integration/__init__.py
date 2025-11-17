"""ITS-Hub integration for LiteLLM - Inference-Time Scaling algorithms."""

from __future__ import annotations

import sys
from typing import Any, Union

# Check Python version first
if sys.version_info < (3, 10):
    raise ImportError(
        "ITS algorithms require Python 3.10+. "
        f"Current Python version: {sys.version_info.major}.{sys.version_info.minor}"
    )

try:
    from its_hub import BestOfN, SelfConsistency
    from its_hub.reward_models import LLMJudge
    from its_hub.types import ChatMessage
except ImportError as e:
    raise ImportError(
        "its-hub is required for ITS algorithms. "
        "Install with: pip install its-hub\n"
        f"Error: {e}"
    )

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Message,
    ModelResponse,
    Usage,
)

from .adapter import LiteLLMLanguageModel

__all__ = ["apply_its_algorithm", "LiteLLMLanguageModel"]


async def apply_its_algorithm(
    model: str,
    messages: list[dict],
    algorithm: str,
    budget: int = 5,
    **litellm_kwargs: Any,
) -> Union[ModelResponse, CustomStreamWrapper]:
    """
    Apply its-hub inference-time scaling algorithm using LiteLLM as the backend.

    This function enables Self-Consistency and Best-of-N algorithms across all
    100+ LiteLLM-supported providers by wrapping litellm.acompletion.

    Args:
        model: LiteLLM model name (e.g., "gpt-4", "anthropic/claude-3-5-sonnet")
        messages: Chat messages in OpenAI format [{"role": "user", "content": "..."}]
        algorithm: Algorithm to use - "self-consistency" or "best-of-n"
        budget: Number of generations (default: 5)
        **litellm_kwargs: Additional parameters passed to litellm.acompletion
                         (temperature, api_key, max_tokens, etc.)

                         Best-of-N specific parameters:
                         - judge_model: Model to use for judging (default: same as generation model)
                         - judge_prompt: Custom judge prompt template with {conversation} placeholder
                         - judge_temperature: Temperature for judge model (default: None)
                         - judge_fallback_score: Fallback score if parsing fails (default: 5.0)

    Returns:
        ModelResponse: When stream=False (default), returns complete response with best/majority result
        CustomStreamWrapper: When stream=True, returns pseudo-streaming wrapper that yields the result

    Raises:
        ValueError: If algorithm is unknown or budget < 1
        ImportError: If its-hub is not installed

    Examples:
        Self-Consistency:
        >>> response = await apply_its_algorithm(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "What is 15 * 24?"}],
        ...     algorithm="self-consistency",
        ...     budget=5,
        ...     temperature=0.7
        ... )
        >>> print(response.choices[0].message.content)  # "360"
        >>> print(response.its_vote_counts)  # {"360": 4, "350": 1}

        Best-of-N with custom judge:
        >>> response = await apply_its_algorithm(
        ...     model="gpt-4o",
        ...     messages=[{"role": "user", "content": "Write a sorting function"}],
        ...     algorithm="best-of-n",
        ...     budget=5,
        ...     judge_model="gpt-4o-mini",  # Cheaper model for judging
        ...     judge_temperature=0.3,
        ... )
        >>> print(response.its_scores)  # [9.5, 8.0, 10.0, 7.5, 9.0]
    """
    # Validate parameters
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")

    # Extract logging object from kwargs if present
    litellm_logging_obj = litellm_kwargs.pop("litellm_logging_obj", None)

    # Handle streaming: Save flag and disable for internal ITS processing
    # ITS requires complete responses, so we'll do pseudo-streaming at the end
    stream_requested = litellm_kwargs.pop("stream", False)
    stream_options = litellm_kwargs.pop("stream_options", None)

    # Extract tools and tool_choice for passing to ainfer
    tools = litellm_kwargs.pop("tools", None)
    tool_choice = litellm_kwargs.pop("tool_choice", None)

    # Extract judge-specific parameters for best-of-n
    judge_model = litellm_kwargs.pop("judge_model", None)
    judge_prompt = litellm_kwargs.pop("judge_prompt", None)
    judge_temperature = litellm_kwargs.pop("judge_temperature", None)
    judge_fallback_score = litellm_kwargs.pop("judge_fallback_score", 5.0)

    # Create LiteLLM wrapper for its-hub (generation model)
    lm = LiteLLMLanguageModel(model=model, **litellm_kwargs)

    # Convert dict messages to ChatMessage objects for its-hub
    chat_messages = [
        ChatMessage(
            role=msg["role"],
            content=msg.get("content"),
            tool_calls=msg.get("tool_calls"),
            tool_call_id=msg.get("tool_call_id"),
        )
        for msg in messages
    ]

    # Select and initialize algorithm
    if algorithm == "self-consistency":
        alg = SelfConsistency()
    elif algorithm == "best-of-n":
        # Create judge model (can be different from generation model)
        if judge_model is not None:
            # Use separate model for judging
            judge_kwargs = {}
            if judge_temperature is not None:
                judge_kwargs["temperature"] = judge_temperature
            judge_lm = LiteLLMLanguageModel(model=judge_model, **judge_kwargs)
        else:
            # Use same model for generation and judging
            judge_lm = lm

        # Create LLM judge with optional custom prompt
        judge = LLMJudge(
            lm=judge_lm,
            judge_prompt=judge_prompt,
            fallback_score=judge_fallback_score,
        )
        alg = BestOfN(judge)
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Supported algorithms: 'self-consistency', 'best-of-n'"
        )

    # Run algorithm with return_response_only=False to get full result object
    result = await alg.ainfer(
        lm,
        chat_messages,
        budget,
        return_response_only=False,
        tools=tools,
        tool_choice=tool_choice,
    )

    # Convert its-hub result to LiteLLM ModelResponse
    response = _convert_to_model_response(result, model, algorithm, budget)

    # Add ITS metadata to response (will be merged with litellm's metadata by wrapper)
    # Store in a way that won't be overwritten
    response.its_algorithm = algorithm
    response.its_budget = budget

    # Add algorithm-specific metadata as attributes
    if algorithm == "self-consistency" and hasattr(result, "response_counts"):
        response.its_vote_counts = dict(result.response_counts)
        response.its_selected_index = result.selected_index
        response.its_total_responses = len(result.responses)
    elif algorithm == "best-of-n" and hasattr(result, "scores"):
        response.its_scores = result.scores
        response.its_selected_index = result.selected_index
        response.its_total_responses = len(result.responses)

    # Add ITS metadata to model_call_details for logging
    # This ensures the metadata is captured in logs even if there's no logging_obj
    if litellm_logging_obj is not None:
        # Add ITS metadata from _hidden_params to model_call_details
        its_metadata = {}
        if hasattr(response, "_hidden_params") and response._hidden_params:
            for key in ["its_algorithm", "its_budget", "vote_counts", "scores", "selected_index", "total_responses"]:
                if key in response._hidden_params:
                    its_metadata[key] = response._hidden_params[key]

        if its_metadata:
            litellm_logging_obj.model_call_details["its_metadata"] = its_metadata

            # Also add to litellm_params for visibility in all logging
            if "litellm_params" not in litellm_logging_obj.model_call_details:
                litellm_logging_obj.model_call_details["litellm_params"] = {}
            litellm_logging_obj.model_call_details["litellm_params"]["its_metadata"] = its_metadata

    # Log ITS metadata to console (always visible in proxy logs)
    import sys
    print(f"[ITS] Applied {algorithm} with budget={budget}, metadata: {response._hidden_params}", file=sys.stderr, flush=True)

    # Handle pseudo-streaming: Convert complete response to streaming format
    if stream_requested:
        import datetime

        # Create a mock iterator that yields the complete response as a single chunk
        mock_iterator = MockResponseIterator(model_response=response)

        # Use existing logging object if provided, otherwise create a minimal one
        if litellm_logging_obj is not None:
            logging_obj = litellm_logging_obj
        else:
            # Create a minimal logging object for the stream wrapper
            # We need this for the CustomStreamWrapper, but we don't need full logging
            logging_obj = LiteLLMLoggingObj(
                model=model,
                messages=messages,
                stream=True,
                call_type="acompletion",
                start_time=datetime.datetime.now(),  # Use datetime object, not float
                litellm_call_id="its-pseudo-stream",
                function_id="its-pseudo-stream",
            )

            # Add call_type to model_call_details for logging callbacks
            # Normally this is done in update_environment_variables(), but we're creating a minimal logging object
            logging_obj.model_call_details["call_type"] = "acompletion"

            # Add ITS metadata to model_call_details
            its_metadata = {}
            if hasattr(response, "_hidden_params") and response._hidden_params:
                for key in ["its_algorithm", "its_budget", "vote_counts", "scores", "selected_index", "total_responses"]:
                    if key in response._hidden_params:
                        its_metadata[key] = response._hidden_params[key]

            if its_metadata:
                logging_obj.model_call_details["its_metadata"] = its_metadata

                # Also add to litellm_params for visibility
                if "litellm_params" not in logging_obj.model_call_details:
                    logging_obj.model_call_details["litellm_params"] = {}
                logging_obj.model_call_details["litellm_params"]["its_metadata"] = its_metadata

        # Wrap in CustomStreamWrapper to return proper streaming response
        stream_wrapper = CustomStreamWrapper(
            completion_stream=mock_iterator,
            model=model,
            custom_llm_provider="openai",  # Use openai format
            logging_obj=logging_obj,
            stream_options=stream_options,
        )

        return stream_wrapper

    return response


def _convert_to_model_response(
    its_result: Any,
    model: str,
    algorithm: str,
    budget: int,
) -> ModelResponse:
    """
    Convert its-hub result object to LiteLLM ModelResponse.

    Args:
        its_result: Result object from its-hub algorithm (SelfConsistencyResult or BestOfNResult)
        model: Model name used
        algorithm: Algorithm name
        budget: Budget used

    Returns:
        ModelResponse with the selected response in choices[0]
    """
    # Extract the selected response from its-hub result
    selected_response = its_result.the_one

    # Convert tool_calls from dicts to ChatCompletionMessageToolCall objects if present
    tool_calls = None
    if "tool_calls" in selected_response and selected_response["tool_calls"] is not None:
        tool_calls = [
            ChatCompletionMessageToolCall(**tc) if isinstance(tc, dict) else tc
            for tc in selected_response["tool_calls"]
        ]

    # Create Message object from selected response
    message = Message(
        role=selected_response.get("role", "assistant"),
        content=selected_response.get("content"),
        tool_calls=tool_calls,
    )

    # Create Choices object
    choice = Choices(
        finish_reason="stop",
        index=0,
        message=message,
    )

    # Build hidden_params with ITS metadata
    hidden_params = {
        "its_algorithm": algorithm,
        "its_budget": budget,
    }

    # Add algorithm-specific metadata
    if algorithm == "self-consistency":
        # SelfConsistencyResult has response_counts attribute
        if hasattr(its_result, "response_counts"):
            # Convert Counter to dict for serialization
            hidden_params["vote_counts"] = dict(its_result.response_counts)
            hidden_params["selected_index"] = its_result.selected_index
            hidden_params["total_responses"] = len(its_result.responses)
    elif algorithm == "best-of-n":
        # BestOfNResult has scores and selected_index
        if hasattr(its_result, "scores"):
            hidden_params["scores"] = its_result.scores
            hidden_params["selected_index"] = its_result.selected_index
            hidden_params["total_responses"] = len(its_result.responses)

    # Create ModelResponse
    # Note: We don't have accurate usage info since its-hub doesn't track tokens
    # Users should calculate N * single_call_tokens if needed
    response = ModelResponse(
        choices=[choice],
        model=model,
    )

    # Set _hidden_params as an attribute (ModelResponse constructor may not preserve it)
    response._hidden_params = hidden_params

    return response
