"""
Majority Voting Algorithm for LiteLLM Router

This module implements a majority voting mechanism where multiple concurrent
calls are made to the same LLM model, and the most common response is returned.

Usage in config.yaml:
    model_list:
      - model_name: gpt-4-with-voting
        litellm_params:
          model: gpt-4
          algorithm: majority-voting
          budget: 8
"""

import asyncio
from collections import Counter
from typing import Any, Dict, List, Optional, Union

import litellm
from litellm import ModelResponse
from litellm.litellm_core_utils.core_helpers import verbose_logger
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.utils import get_response_string


async def majority_voting_completion(
    router_instance: Any,
    model: str,
    messages: List[Dict[str, str]],
    budget: int,
    **kwargs,
) -> Union[ModelResponse, CustomStreamWrapper]:
    """
    Make `budget` concurrent calls to the same model and return the majority response.

    Args:
        router_instance: The Router instance (self from Router._acompletion)
        model: Model name to call
        messages: Chat messages
        budget: Number of concurrent calls to make
        **kwargs: Additional parameters for completion

    Returns:
        ModelResponse with the most common response content

    Raises:
        ValueError: If budget < 1 or if streaming is enabled
        Exception: If all calls fail
    """
    # Validate budget
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")

    # Check if streaming is enabled
    if kwargs.get("stream", False):
        raise ValueError("Majority voting does not support streaming")

    # Log start of majority voting
    verbose_logger.info(
        f"🗳️  MAJORITY VOTING: Starting {budget} concurrent calls to model '{model}'"
    )

    # Remove algorithm and budget params to avoid passing to litellm
    cleaned_kwargs = kwargs.copy()
    cleaned_kwargs.pop("algorithm", None)
    cleaned_kwargs.pop("budget", None)

    # Create wrapper function to catch exceptions
    async def _safe_completion(
        call_index: int,
    ) -> Union[ModelResponse, Exception]:
        """
        Wrapper around completion that catches exceptions and returns them.
        """
        try:
            # Use the same deployment flow as normal _acompletion
            # Get deployment
            deployment = await router_instance.async_get_available_deployment(
                model=model,
                messages=messages,
                specific_deployment=cleaned_kwargs.get("specific_deployment", None),
                request_kwargs=cleaned_kwargs,
            )

            # Update kwargs with deployment params
            router_instance._update_kwargs_with_deployment(
                deployment=deployment, kwargs=cleaned_kwargs
            )
            data = deployment["litellm_params"]
            model_name = data["model"]

            # Get model client
            model_client = router_instance._get_async_openai_model_client(
                deployment=deployment,
                kwargs=cleaned_kwargs,
            )

            # Track metrics
            router_instance.total_calls[model_name] += 1

            # Prepare input kwargs - remove algorithm and budget from data
            data_cleaned = {k: v for k, v in data.items() if k not in ["algorithm", "budget"]}

            input_kwargs = {
                **data_cleaned,
                "messages": messages,
                "caching": router_instance.cache_responses,
                "client": model_client,
                **cleaned_kwargs,
            }

            # Make the completion call
            response = await litellm.acompletion(**input_kwargs)
            return response

        except asyncio.CancelledError:
            raise
        except Exception as e:
            return e

    # Create budget number of concurrent tasks
    tasks = [
        asyncio.create_task(_safe_completion(call_index=i))
        for i in range(budget)
    ]

    # Gather all responses
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter successful responses
    valid_responses = [
        r for r in responses if isinstance(r, ModelResponse)
    ]

    # Log results summary
    failed_count = budget - len(valid_responses)
    verbose_logger.info(
        f"🗳️  MAJORITY VOTING: Completed - {len(valid_responses)}/{budget} calls succeeded, "
        f"{failed_count} failed"
    )

    if not valid_responses:
        # All calls failed - collect error messages
        error_messages = []
        for r in responses:
            if isinstance(r, Exception):
                error_messages.append(str(r))
        raise Exception(
            f"All {budget} majority voting calls failed. Errors: {error_messages}"
        )

    # Find majority response and get vote counts
    majority_response, vote_counts, winning_content = _find_majority_response(valid_responses)

    # Log voting results
    verbose_logger.info(
        f"🗳️  MAJORITY VOTING: Vote distribution:"
    )
    for response_content, count in sorted(vote_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(valid_responses)) * 100
        # Truncate long responses for logging
        content_preview = response_content[:100] + "..." if len(response_content) > 100 else response_content
        verbose_logger.info(
            f"   {'✓' if response_content == winning_content else ' '} {count}/{len(valid_responses)} ({percentage:.1f}%): {content_preview}"
        )

    verbose_logger.info(
        f"🗳️  MAJORITY VOTING: Selected response with {vote_counts[winning_content]}/{len(valid_responses)} votes"
    )

    # Add metadata to indicate this was a majority voting response
    if not hasattr(majority_response, "_hidden_params"):
        majority_response._hidden_params = {}
    majority_response._hidden_params["majority_voting"] = True
    majority_response._hidden_params["majority_voting_budget"] = budget
    majority_response._hidden_params["majority_voting_successful_calls"] = len(
        valid_responses
    )
    majority_response._hidden_params["majority_voting_failed_calls"] = (
        budget - len(valid_responses)
    )
    majority_response._hidden_params["majority_voting_response_counts"] = vote_counts
    majority_response._hidden_params["majority_voting_winning_response"] = winning_content

    return majority_response


def _find_majority_response(
    responses: List[ModelResponse],
) -> tuple[ModelResponse, dict, str]:
    """
    Find the response with the most common content.

    Args:
        responses: List of ModelResponse objects

    Returns:
        Tuple of (majority_response, vote_counts, winning_content)
        - majority_response: The ModelResponse with the most common content
        - vote_counts: Dict mapping content to count (e.g., {"Answer A": 5, "Answer B": 3})
        - winning_content: The winning response content string
    """
    # Extract response content from each response
    response_contents = []
    response_map = {}  # Map content to first occurrence of that response

    for response in responses:
        content = get_response_string(response)
        response_contents.append(content)
        if content not in response_map:
            response_map[content] = response

    # Count occurrences
    content_counter = Counter(response_contents)

    # Get most common content
    most_common_content, most_common_count = content_counter.most_common(1)[0]

    # Return the first response with that content, plus metadata
    majority_response = response_map[most_common_content]
    vote_counts = dict(content_counter)

    return majority_response, vote_counts, most_common_content
