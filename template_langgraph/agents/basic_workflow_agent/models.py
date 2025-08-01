from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentInput(BaseModel):
    request: str = Field(..., description="ユーザーからのリクエスト")


class AgentOutput(BaseModel):
    response: str = Field(..., description="エージェントの応答")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
