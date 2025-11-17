#!/usr/bin/env python3
"""Test pseudo-streaming for ITS algorithms."""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import litellm


async def test_streaming():
    """Test that ITS algorithms work with stream=True."""
    print("Testing ITS pseudo-streaming...")
    print("=" * 60)

    # Test 1: Self-consistency with streaming
    print("\nTest 1: Self-consistency with stream=True")
    print("-" * 60)

    response_stream = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2+2? Just answer with the number."}],
        algorithm="self-consistency",
        budget=3,
        temperature=0.7,
        stream=True,  # Enable streaming
    )

    print(f"Response type: {type(response_stream)}")
    print(f"Response object: {response_stream}")

    # Iterate through the stream
    full_content = ""
    chunk_count = 0
    async for chunk in response_stream:
        chunk_count += 1
        print(f"Chunk {chunk_count}: {chunk}")
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                full_content += delta.content
                print(f"  Content: {delta.content}")
            if hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason:
                print(f"  Finish reason: {chunk.choices[0].finish_reason}")

    print(f"\nTotal chunks: {chunk_count}")
    print(f"Full content: {full_content}")

    # Test 2: Non-streaming for comparison
    print("\n" + "=" * 60)
    print("Test 2: Self-consistency without streaming (for comparison)")
    print("-" * 60)

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2+2? Just answer with the number."}],
        algorithm="self-consistency",
        budget=3,
        temperature=0.7,
        stream=False,  # No streaming
    )

    print(f"Response type: {type(response)}")
    print(f"Content: {response.choices[0].message.content}")
    print(f"ITS metadata:")
    if hasattr(response, 'its_algorithm'):
        print(f"  Algorithm: {response.its_algorithm}")
    if hasattr(response, 'its_budget'):
        print(f"  Budget: {response.its_budget}")
    if hasattr(response, 'its_vote_counts'):
        print(f"  Vote counts: {response.its_vote_counts}")

    print("\n" + "=" * 60)
    print("✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(test_streaming())
