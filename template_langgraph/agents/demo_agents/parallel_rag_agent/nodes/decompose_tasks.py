import json
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools.base import BaseTool
from langgraph.types import Command, Send

from template_langgraph.agents.demo_agents.parallel_rag_agent.models import (
    Task,
    Tasks,
)
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class DecomposeTasks:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
    ):
        self.llm = llm
        self.tools = tools

    def __call__(self, state: dict) -> Command[Literal["run_task"]]:
        query = state.get("query", "")
        response: AIMessage = self.llm.bind_tools(tools=self.tools).invoke(query)

        logger.info(f"{response}, {type(response)}")
        gotos = []
        tasks_list: list[Task] = []
        for tool_call in response.tool_calls:
            logger.info(f"name={tool_call['name']}, args={tool_call['args']}")
            args = tool_call.get("args", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            task = Task(
                id=tool_call["id"],
                tool_name=tool_call["name"],
                tool_args=args,
            )
            tasks_list.append(task)
            gotos.append(
                Send(
                    "run_task",
                    {
                        "task": task,
                        "query": query,
                    },
                )
            )

        return Command(
            goto=gotos,
            update={"tasks": Tasks(tasks=tasks_list)},
        )
