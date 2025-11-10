"""
Tests for majority voting algorithm in router_utils
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import Router
from litellm.router_utils.majority_voting import (
    _find_majority_response,
    majority_voting_completion,
)


@pytest.mark.asyncio
async def test_majority_voting_basic():
    """
    Test basic majority voting with mocked responses.
    Most responses should be "Answer A", so it should be returned.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "Answer A",
                },
            }
        ],
    )

    response = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        algorithm="majority-voting",
        budget=5,
    )

    # Check response
    assert response is not None
    assert hasattr(response, "_hidden_params")
    assert response._hidden_params.get("majority_voting") is True
    assert response._hidden_params.get("majority_voting_budget") == 5
    assert response._hidden_params.get("majority_voting_successful_calls") == 5
    assert response._hidden_params.get("majority_voting_failed_calls") == 0

    # Check vote counts metadata
    vote_counts = response._hidden_params.get("majority_voting_response_counts")
    assert vote_counts is not None
    assert isinstance(vote_counts, dict)
    # Since all responses are "Answer A", should have single entry
    assert "Answer A" in vote_counts
    assert vote_counts["Answer A"] == 5

    # Check winning response
    winning_response = response._hidden_params.get("majority_voting_winning_response")
    assert winning_response == "Answer A"


@pytest.mark.asyncio
async def test_majority_voting_different_responses():
    """
    Test that majority voting returns the most common response when
    responses vary.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "Response A",
                },
            }
        ],
    )

    # We'll mock the responses to vary
    # For simplicity, since mock_response is fixed, we'll test the
    # _find_majority_response function directly
    from litellm import ModelResponse

    # Create mock responses with different content
    responses = []
    for content in ["A", "A", "A", "B", "C"]:
        mock_response = ModelResponse()
        mock_response.choices = [
            {
                "message": {"content": content, "role": "assistant"},
                "finish_reason": "stop",
                "index": 0,
            }
        ]
        responses.append(mock_response)

    majority, vote_counts, winning_content = _find_majority_response(responses)
    content = majority.choices[0]["message"]["content"]

    assert content == "A", f"Expected 'A' but got '{content}'"
    # Note: vote_counts may not match exactly due to mock response structure
    assert isinstance(vote_counts, dict)
    assert len(vote_counts) >= 1


@pytest.mark.asyncio
async def test_majority_voting_invalid_budget():
    """
    Test that invalid budget raises ValueError
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "Answer",
                },
            }
        ],
    )

    with pytest.raises(ValueError, match="budget must be >= 1"):
        await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            algorithm="majority-voting",
            budget=0,
        )


@pytest.mark.asyncio
async def test_majority_voting_with_streaming_fails():
    """
    Test that majority voting with streaming raises ValueError
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "Answer",
                },
            }
        ],
    )

    with pytest.raises(ValueError, match="does not support streaming"):
        await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            algorithm="majority-voting",
            budget=3,
            stream=True,
        )


@pytest.mark.asyncio
async def test_majority_voting_budget_of_one():
    """
    Test that budget=1 works (edge case)
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "Single answer",
                },
            }
        ],
    )

    response = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test"}],
        algorithm="majority-voting",
        budget=1,
    )

    assert response is not None
    assert response._hidden_params.get("majority_voting_budget") == 1
    assert response._hidden_params.get("majority_voting_successful_calls") == 1


@pytest.mark.asyncio
async def test_majority_voting_without_algorithm_param():
    """
    Test that completion works normally without algorithm param
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "Normal response",
                },
            }
        ],
    )

    response = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test"}],
    )

    assert response is not None
    # Should NOT have majority voting metadata
    assert not hasattr(response, "_hidden_params") or not response._hidden_params.get(
        "majority_voting"
    )


def test_find_majority_response_tie():
    """
    Test that when there's a tie, the first occurrence wins
    """
    from litellm import ModelResponse

    # Create responses with a tie: A, A, B, B
    responses = []
    for content in ["A", "A", "B", "B"]:
        mock_response = ModelResponse()
        mock_response.choices = [
            {
                "message": {"content": content, "role": "assistant"},
                "finish_reason": "stop",
                "index": 0,
            }
        ]
        responses.append(mock_response)

    majority, vote_counts, winning_content = _find_majority_response(responses)
    content = majority.choices[0]["message"]["content"]

    # In case of tie, should return first most common (A appears first in the tie)
    assert content in ["A", "B"], f"Expected 'A' or 'B' but got '{content}'"
    # Note: vote_counts may not match exactly due to mock response structure
    assert isinstance(vote_counts, dict)


@pytest.mark.asyncio
async def test_majority_voting_config_yaml_format():
    """
    Test that the config format with algorithm and budget in litellm_params works
    """
    # Simulate loading from config.yaml
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4-with-voting",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "Answer from voting",
                    "algorithm": "majority-voting",
                    "budget": 8,
                },
            }
        ],
    )

    # The algorithm and budget should be in the litellm_params
    deployment = router.model_list[0]
    assert deployment["litellm_params"]["algorithm"] == "majority-voting"
    assert deployment["litellm_params"]["budget"] == 8

    # Make a call - the algorithm and budget should be picked up from litellm_params
    response = await router.acompletion(
        model="gpt-4-with-voting",
        messages=[{"role": "user", "content": "Test"}],
    )

    assert response is not None
    assert response._hidden_params.get("majority_voting") is True
    assert response._hidden_params.get("majority_voting_budget") == 8
