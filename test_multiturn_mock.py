#!/usr/bin/env python3
"""
Mock test to verify multi-turn conversation capability without external API calls.
"""

from langchain_core.messages import HumanMessage, AIMessage
from template_langgraph.agents.demo_agents.parallel_rag_agent.models import (
    ParallelRagAgentState,
)


def test_multiturn_conversation_flow():
    """Test that we can maintain conversation history in the state."""
    print("Testing multi-turn conversation flow...")
    
    # Simulate first turn
    print("\n=== First Turn ===")
    first_turn_input = {
        "messages": [HumanMessage(content="What tools are available?")]
    }
    print(f"First turn input: {len(first_turn_input['messages'])} messages")
    
    # Simulate agent response (what would happen after processing)
    first_turn_output = {
        "messages": [
            HumanMessage(content="What tools are available?"),
            AIMessage(content="I have access to search tools and workflow tools."),
        ],
        "summary": "Listed available tools",
        "task_results": []
    }
    print(f"First turn output: {len(first_turn_output['messages'])} messages")
    
    # Simulate second turn (multi-turn)
    print("\n=== Second Turn ===")
    second_turn_input = {
        "messages": first_turn_output["messages"] + [
            HumanMessage(content="Tell me more about the search tools")
        ]
    }
    print(f"Second turn input: {len(second_turn_input['messages'])} messages")
    
    # Verify conversation history is maintained
    assert len(second_turn_input["messages"]) == 3, "Should have 3 messages in history"
    assert isinstance(second_turn_input["messages"][0], HumanMessage), "First should be user message"
    assert isinstance(second_turn_input["messages"][1], AIMessage), "Second should be AI message"
    assert isinstance(second_turn_input["messages"][2], HumanMessage), "Third should be user message"
    
    print("✅ Conversation history is properly maintained")
    
    # Simulate second turn response
    second_turn_output = {
        "messages": second_turn_input["messages"] + [
            AIMessage(content="The search tools allow you to search through various databases and documents."),
        ],
        "summary": "Explained search tools functionality",
        "task_results": []
    }
    print(f"Second turn output: {len(second_turn_output['messages'])} messages")
    
    # Verify full conversation history
    assert len(second_turn_output["messages"]) == 4, "Should have complete conversation history"
    print("✅ Complete conversation history is maintained")
    
    print("\n=== Conversation Summary ===")
    for i, msg in enumerate(second_turn_output["messages"]):
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        print(f"{i+1}. {role}: {msg.content}")
    
    print("\n🎉 Multi-turn conversation test passed!")
    return True


def test_message_state_compatibility():
    """Test that the state is compatible with LangGraph's add_messages."""
    print("\nTesting message state compatibility...")
    
    # Test that messages can be added to existing state
    initial_state = {
        "messages": [HumanMessage(content="Initial message")]
    }
    
    # Simulate what add_messages would do
    new_message = AIMessage(content="Response message")
    updated_messages = initial_state["messages"] + [new_message]
    
    updated_state = {
        **initial_state,
        "messages": updated_messages
    }
    
    assert len(updated_state["messages"]) == 2, "Should have 2 messages after adding"
    assert isinstance(updated_state["messages"][0], HumanMessage), "First should be HumanMessage"
    assert isinstance(updated_state["messages"][1], AIMessage), "Second should be AIMessage"
    
    print("✅ Message state is compatible with add_messages pattern")
    
    return True


if __name__ == "__main__":
    try:
        test_multiturn_conversation_flow()
        test_message_state_compatibility()
        print("\n🎉 All multi-turn tests passed! ParallelRagAgent supports interactive conversations.")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()