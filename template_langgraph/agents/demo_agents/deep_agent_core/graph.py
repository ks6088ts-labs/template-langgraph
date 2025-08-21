from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from deepagents.model import get_default_model
from deepagents.state import DeepAgentState
from deepagents.sub_agent import SubAgent, _create_task_tool
from deepagents.tools import edit_file, ls, read_file, write_file, write_todos
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.types import Checkpointer

StateSchema = TypeVar("StateSchema", bound=DeepAgentState)
StateSchemaType = type[StateSchema]

base_prompt = """You have access to a number of standard tools

## `write_todos`

You have access to the `write_todos` tools to help you manage and plan tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.
These tools are also EXTREMELY helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable.

It is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.
## `task`

- When doing web search, prefer to use the `task` tool in order to reduce context usage."""


def create_deep_agent(
    tools: Sequence[BaseTool | Callable | dict[str, Any]],
    instructions: str,
    model: str | LanguageModelLike | None = None,
    subagents: list[SubAgent] = None,
    state_schema: StateSchemaType | None = None,
    config_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
):
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    and then four file editing tools: write_file, ls, read_file, edit_file.

    Args:
        tools: The additional tools the agent should have access to.
        instructions: The additional instructions the agent should have. Will go in
            the system prompt.
        model: The model to use.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
        state_schema: The schema of the deep agent. Should subclass from DeepAgentState
        config_schema: The schema of the deep agent.
        checkpointer: Optional checkpointer for persisting agent state between runs.
    """
    prompt = instructions + base_prompt
    built_in_tools = [write_todos, write_file, read_file, ls, edit_file]
    if model is None:
        model = get_default_model()
    state_schema = state_schema or DeepAgentState
    task_tool = _create_task_tool(list(tools) + built_in_tools, instructions, subagents or [], model, state_schema)
    all_tools = built_in_tools + list(tools) + [task_tool]
    return create_react_agent(
        model,
        prompt=prompt,
        tools=all_tools,
        state_schema=state_schema,
        config_schema=config_schema,
        checkpointer=checkpointer,
    )
