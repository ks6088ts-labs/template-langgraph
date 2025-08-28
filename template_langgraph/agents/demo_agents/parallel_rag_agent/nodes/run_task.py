import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools.base import BaseTool

from template_langgraph.agents.demo_agents.parallel_rag_agent.models import (
    Task,
    TaskResult,
)
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class RunTask:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
    ):
        self.llm = llm
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: dict) -> dict:
        logger.info(f"Running state... {state}")
        task: Task = state.get("task", None)
        query: str = state.get("query", None)
        messages = state.get("messages", [])
        logger.info(f"Task: {task.model_dump_json(indent=2)}")

        try:
            observation = self.tools_by_name[task.tool_name].invoke(task.tool_args)
        except Exception as e:
            logger.error(f"Error occurred while invoking tools: {e}")
            observation = {"error": str(e)}

        # Build context for LLM using conversation history
        context_messages = messages.copy() if messages else []
        
        # Add system message to explain the context
        system_message = SystemMessage(
            content=f"You are processing a task as part of a conversation. "
            f"Task: {task.tool_name} with arguments {task.tool_args}. "
            f"Based on the conversation history and the following observation from the tool, "
            f"provide a helpful response that answers the user's question."
        )
        context_messages.insert(0, system_message)
        
        # Add the tool observation
        context_messages.append(
            HumanMessage(
                content=f"Tool observation: {json.dumps(observation.__str__(), ensure_ascii=False)}"
            )
        )

        result = self.llm.invoke(input=context_messages)

        logger.info(f"LLM response: {result.model_dump_json(indent=2)}, type: {type(result)}")

        result = TaskResult(
            task=task,
            result_code=0,
            message=result.content,
        )

        return {
            "task_results": [result],
        }
