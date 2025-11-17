#!/usr/bin/env python3
"""
Minimal LangGraph agent with calculator tool.
Based on solid React agent pattern from volcengine/verl.

Usage with ITS-configured LiteLLM proxy:
----------------------------------------

1. Standard model (no ITS):
   python test_langgraph_agent.py \
       --endpoint http://localhost:4000/v1 \
       --model gpt-4o-mini \
       --problem "What is 15 * 24?"

2. Self-consistency model (pre-configured in proxy YAML):
   python test_langgraph_agent.py \
       --endpoint http://localhost:4000/v1 \
       --model gpt-4o-mini-sc \
       --problem "What is 15 * 24?"

3. Best-of-N model (pre-configured with custom judge):
   python test_langgraph_agent.py \
       --endpoint http://localhost:4000/v1 \
       --model gpt-4o-bon \
       --problem "Solve: 123 + 456 * 789"

Note: ITS parameters (algorithm, budget, judge) are configured in the
proxy's YAML config file, not passed in the request. This ensures
compatibility with LangGraph and other frameworks that have strict
request schemas.
"""

import argparse
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from typing_extensions import Annotated


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return f"Error: Invalid characters"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def call_model(state: AgentState):
    """Call the LLM with tools."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    
    # Show tool calls if present
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            print(f"Tool call: {tc}")
    
    return {"messages": [response]}


def should_continue(state: AgentState):
    """Decide whether to continue with tools or end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"


def create_agent(endpoint: str, model: str = "gpt-4o-mini"):
    """Create the minimal LangGraph agent."""
    global llm_with_tools

    # Setup LLM with pre-configured ITS model from proxy
    # ITS params (algorithm, budget, judge, etc.) are configured in proxy YAML
    api_key = "not-needed" if "localhost" in endpoint else os.getenv("OPENAI_API_KEY")

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=endpoint,
        model=model,  # Use pre-configured ITS model (e.g., gpt-4o-mini-sc, gpt-4o-bon)
    )
    
    llm_with_tools = llm.bind_tools([calculator])
    
    # Create tools node
    tools = [calculator]
    tool_node = ToolNode(tools)
    
    # Create graph like volcengine/verl pattern
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph agent with ITS-configured LiteLLM proxy"
    )
    parser.add_argument("--endpoint", default="http://localhost:4000/v1", help="LiteLLM proxy endpoint URL")
    parser.add_argument("--problem", default="What is 15 * 24?", help="Math problem to solve")
    parser.add_argument("--model", required=True, help="Model name from proxy config (e.g., gpt-4o-mini-sc for self-consistency, gpt-4o-bon for best-of-n)")
    args = parser.parse_args()

    # Create agent with pre-configured ITS model
    agent = create_agent(args.endpoint, args.model)
    
    # Run agent
    result = agent.invoke({
        "messages": [
            SystemMessage(content="Use calculator tool for math. Put final answer in \\boxed{}."),
            HumanMessage(content=args.problem)
        ]
    })
    
    # Print final response
    if result["messages"]:
        final_msg = result["messages"][-1]
        if hasattr(final_msg, 'content') and final_msg.content:
            print(final_msg.content)


if __name__ == "__main__":
    main()