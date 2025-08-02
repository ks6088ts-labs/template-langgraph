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


class AgentInput(BaseModel):
    request: str = Field(..., description="ユーザーからのリクエスト")


class AgentOutput(BaseModel):
    response: str = Field(..., description="エージェントの応答")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
