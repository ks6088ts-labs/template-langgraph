#!/usr/bin/env python3
"""
Test script for ParallelRagAgent with message state support.
This tests multi-turn conversation capability.
"""

from langchain_core.messages import HumanMessage, AIMessage
from template_langgraph.agents.demo_agents.parallel_rag_agent.agent import ParallelRagAgent
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.tools.common import get_default_tools


def test_parallel_rag_agent_messages():
    """Test that the ParallelRagAgent works with message state."""
    print("Testing ParallelRagAgent with message state...")
    
    # Create agent with a few tools for testing
    tools = get_default_tools()[:3]  # Limit tools for faster testing
    agent = ParallelRagAgent(
        llm=AzureOpenAiWrapper().chat_model,
        tools=tools,
    )
    graph = agent.create_graph()
    
    print(f"Using tools: {[tool.name for tool in tools]}")
    
    # Test first turn
    print("\n=== First Turn ===")
    initial_messages = [
        HumanMessage(content="What tools do you have access to?")
    ]
    
    response1 = graph.invoke({
        "messages": initial_messages
    })
    
    print("Response 1 messages:")
    for msg in response1.get("messages", []):
        print(f"- {type(msg).__name__}: {msg.content[:100]}...")
    print(f"Summary: {response1.get('summary', 'N/A')[:100]}...")
    
    # Test second turn (multi-turn conversation)
    print("\n=== Second Turn ===")
    # Include conversation history from first turn
    conversation_messages = response1.get("messages", [])
    conversation_messages.append(
        HumanMessage(content="Can you tell me more about the first tool you mentioned?")
    )
    
    response2 = graph.invoke({
        "messages": conversation_messages
    })
    
    print("Response 2 messages:")
    for msg in response2.get("messages", []):
        print(f"- {type(msg).__name__}: {msg.content[:100]}...")
    print(f"Summary: {response2.get('summary', 'N/A')[:100]}...")
    
    print("\n=== Test Completed Successfully ===")
    return True


if __name__ == "__main__":
    try:
        test_parallel_rag_agent_messages()
        print("✅ All tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()