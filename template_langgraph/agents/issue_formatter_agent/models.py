from collections.abc import Sequence
from enum import Enum
from typing import (
    Annotated,
    TypedDict,
)

from langchain_core.messages import (
    BaseMessage,
)
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class SystemInfo(BaseModel):
    os: str | None = Field(None, description="Operating system of the environment")
    version: str | None = Field(None, description="Version of the environment")


class IssueLabel(Enum):
    """Enum for issue labels."""

    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    DOCUMENTATION = "documentation"
    ENHANCEMENT = "enhancement"
    QUESTION = "question"


class Issue(BaseModel):
    title: str = Field(..., description="Issue title")
    description: str = Field(..., description="Issue description")
    labels: Sequence[IssueLabel] = Field(default_factory=list, description="Labels associated with the issue")
    assignee: str | None = Field(None, description="User assigned to the issue")
    milestone: str | None = Field(None, description="Milestone associated with the issue")
    system_info: SystemInfo | None = Field(None, description="Information about the system")
    steps_to_reproduce: Sequence[str] | None = Field(None, description="Steps to reproduce the issue")
    current_behavior: str | None = Field(None, description="Current behavior when the issue occurs")
    expected_behavior: str | None = Field(None, description="Expected behavior when the issue is resolved")


class AgentInput(BaseModel):
    issue: str = Field(..., description="The issue to be formatted")


class AgentOutput(BaseModel):
    response: str = Field(..., description="The agent's response after formatting the issue")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    issue: Issue
