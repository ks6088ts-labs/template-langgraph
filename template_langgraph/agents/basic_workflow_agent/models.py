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


class Profile(BaseModel):
    first_name: str = Field(..., description="名")
    last_name: str = Field(..., description="姓")
    age: int | None = Field(None, description="年齢")
    address: str | None = Field(None, description="住所")


class AgentInput(BaseModel):
    request: str = Field(..., description="ユーザーからのリクエスト")


class AgentOutput(BaseModel):
    response: str = Field(..., description="エージェントの応答")
    profile: Profile | None = Field(None, description="抽出されたプロファイル情報")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    profile: Profile | None = Field(None, description="抽出されたプロファイル情報")
