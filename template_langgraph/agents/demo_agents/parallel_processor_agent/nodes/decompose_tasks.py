from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command, Send

from template_langgraph.agents.demo_agents.parallel_processor_agent.models import (
    Tasks,
)


class DecomposeTasks:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def __call__(self, state: dict) -> Command[Literal["run_task"]]:
        goal = state.get("goal", "")
        tasks: Tasks = self.llm.with_structured_output(Tasks).invoke(
            input=f"Decompose the following goal into tasks: {goal}",
        )
        gotos = []
        for task in tasks.tasks:
            gotos.append(
                Send(
                    "run_task",
                    {
                        "task": task,
                    },
                )
            )

        return Command(
            goto=gotos,
            update={"tasks": tasks},
        )
