from collections.abc import Sequence
from typing import (
    Annotated,
    TypedDict,
)

from langchain_core.messages import (
    BaseMessage,
)
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Task(BaseModel):
    title: str = Field(..., description="Title of the task")
    description: str = Field(..., description="Description of the task")
    priority: int = Field(..., description="Priority of the task (1-5)")
    due_date: str | None = Field(None, description="Due date of the task (YYYY-MM-DD format)")
    assigned_to: str | None = Field(None, description="Name of the agent assigned to the task")


class AgentInput(BaseModel):
    request: str = Field(..., description="Request from the user")


class AgentOutput(BaseModel):
    response: str = Field(..., description="Response from the agent")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    decomposed_tasks: Sequence[Task]
