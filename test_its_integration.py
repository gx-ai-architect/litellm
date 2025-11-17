"""
Manual test script for ITS-Hub integration with LiteLLM.

This script tests both self-consistency and best-of-n algorithms.
Requires OPENAI_API_KEY to be set in environment.
"""

import asyncio
import os
import sys

# Add litellm to path
sys.path.insert(0, os.path.abspath("."))

import litellm


async def test_self_consistency():
    """Test self-consistency algorithm."""
    print("=" * 60)
    print("Testing Self-Consistency Algorithm")
    print("=" * 60)

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "What is 15 * 24? Just give me the number as your answer.",
            }
        ],
        algorithm="self-consistency",
        budget=5,
        temperature=0.7,
    )

    print(f"\nSelected Answer: {response.choices[0].message.content}")
    print(f"\nITS Metadata:")
    print(f"  Algorithm: {getattr(response, 'its_algorithm', 'NOT FOUND')}")
    print(f"  Budget: {getattr(response, 'its_budget', 'NOT FOUND')}")
    print(f"  Vote Counts: {getattr(response, 'its_vote_counts', {})}")
    print(f"  Selected Index: {getattr(response, 'its_selected_index', 'N/A')}")
    print(f"  Total Responses: {getattr(response, 'its_total_responses', 'N/A')}")
    print()


async def test_best_of_n():
    """Test best-of-n algorithm with same model for judging."""
    print("=" * 60)
    print("Testing Best-of-N Algorithm (Same Model)")
    print("=" * 60)

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Write a one-line Python function that adds two numbers.",
            }
        ],
        algorithm="best-of-n",
        budget=3,
        temperature=0.5,
    )

    print(f"\nSelected Answer: {response.choices[0].message.content}")
    print(f"\nITS Metadata:")
    print(f"  Algorithm: {getattr(response, 'its_algorithm', 'NOT FOUND')}")
    print(f"  Budget: {getattr(response, 'its_budget', 'NOT FOUND')}")
    print(f"  Scores: {getattr(response, 'its_scores', [])}")
    print(f"  Selected Index: {getattr(response, 'its_selected_index', 'N/A')}")
    print(f"  Total Responses: {getattr(response, 'its_total_responses', 'N/A')}")
    print()


async def test_best_of_n_with_custom_judge():
    """Test best-of-n with separate cheaper judge model."""
    print("=" * 60)
    print("Testing Best-of-N with Custom Judge")
    print("=" * 60)

    # Use gpt-4o for generation, gpt-4o-mini for judging (cheaper!)
    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": "Explain what a binary search tree is in one sentence.",
            }
        ],
        algorithm="best-of-n",
        budget=3,
        temperature=0.7,
        # Judge configuration
        judge_model="gpt-4o-mini",  # Use cheaper model for judging
        judge_temperature=0.3,
    )

    print(f"\nGeneration Model: gpt-4o")
    print(f"Judge Model: gpt-4o-mini")
    print(f"\nSelected Answer: {response.choices[0].message.content}")
    print(f"\nITS Metadata:")
    print(f"  Scores: {getattr(response, 'its_scores', [])}")
    print(f"  Selected Index: {getattr(response, 'its_selected_index', 'N/A')}")
    print()


async def main():
    """Run all tests."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set it and try again:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ITS-Hub Integration Manual Test")
    print("=" * 60 + "\n")

    try:
        await test_self_consistency()
        await test_best_of_n()
        await test_best_of_n_with_custom_judge()

        print("=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
