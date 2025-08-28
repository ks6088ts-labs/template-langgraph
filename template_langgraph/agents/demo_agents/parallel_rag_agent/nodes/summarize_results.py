from langchain_core.messages import AIMessage

from template_langgraph.agents.demo_agents.parallel_rag_agent.models import (
    ParallelRagAgentState,
    TaskResult,
)
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class SummarizeResults:
    def __init__(self):
        pass

    def __call__(self, state: ParallelRagAgentState) -> dict:
        logger.info(f"Summarizing results... {state}")
        task_results: list[TaskResult] = state.get("task_results", [])
        
        # Create a comprehensive summary from all task results
        summary_parts = []
        for task_result in task_results:
            summary_parts.append(f"**{task_result['task'].tool_name}**: {task_result['message']}")
        
        # Combine all results into a coherent response
        if summary_parts:
            summary = "\n\n".join(summary_parts)
        else:
            summary = "I wasn't able to find any relevant information for your query."
            
        logger.info(f"Final summary: {summary}")
        
        # Add the final response as an AI message to continue the conversation
        ai_response = AIMessage(content=summary)
        
        return {
            "summary": summary,
            "messages": [ai_response],
        }
