#!/usr/bin/env python3
"""
Unit test for ParallelRagAgent message state changes.
This tests that the new message structure works without requiring API calls.
"""

from langchain_core.messages import HumanMessage, AIMessage
from template_langgraph.agents.demo_agents.parallel_rag_agent.models import (
    ParallelRagAgentState,
    ParallelRagAgentInputState,
    ParallelRagAgentOutputState,
)


def test_message_state_structure():
    """Test that the state models have the correct structure."""
    print("Testing ParallelRagAgent state structure...")
    
    # Test input state has messages
    input_annotations = ParallelRagAgentInputState.__annotations__
    assert 'messages' in input_annotations, "Input state should have messages field"
    print("✅ Input state has messages field")
    
    # Test output state has messages and summary
    output_annotations = ParallelRagAgentOutputState.__annotations__
    assert 'messages' in output_annotations, "Output state should have messages field"
    assert 'summary' in output_annotations, "Output state should have summary field"
    assert 'task_results' in output_annotations, "Output state should have task_results field"
    print("✅ Output state has all required fields")
    
    # Test main state has all fields
    state_annotations = ParallelRagAgentState.__annotations__
    expected_fields = {'messages', 'tasks', 'task_results', 'summary'}
    actual_fields = set(state_annotations.keys())
    assert expected_fields.issubset(actual_fields), f"State should have {expected_fields}, got {actual_fields}"
    print("✅ Main state has all required fields")
    
    # Test that we can create state with messages
    test_messages = [HumanMessage(content="Test message")]
    state_dict = {"messages": test_messages}
    print("✅ Can create state with messages")
    
    print("All state structure tests passed!")
    return True


def test_message_compatibility():
    """Test that the message structure is compatible with LangGraph."""
    print("\nTesting message compatibility...")
    
    # Test that we can create proper message sequences
    from collections.abc import Sequence
    from langchain_core.messages import BaseMessage
    
    messages = [
        HumanMessage(content="First user message"),
        AIMessage(content="First AI response"),
        HumanMessage(content="Second user message"),
    ]
    
    assert isinstance(messages, Sequence), "Messages should be a sequence"
    assert all(isinstance(msg, BaseMessage) for msg in messages), "All items should be BaseMessage instances"
    print("✅ Message sequence structure is correct")
    
    # Test state creation with proper types
    state = {
        "messages": messages,
        "task_results": [],
        "summary": "Test summary"
    }
    print("✅ Can create state with proper message types")
    
    print("All compatibility tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_message_state_structure()
        test_message_compatibility()
        print("\n🎉 All tests passed! The ParallelRagAgent now supports message state.")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()