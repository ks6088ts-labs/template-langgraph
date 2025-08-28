import operator
from collections.abc import Sequence
from typing import (
    Annotated,
    TypedDict,
)

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Task(BaseModel):
    id: str = Field(description="タスクのID")
    tool_name: str = Field(description="タスクのツール名")
    tool_args: dict = Field(description="タスクのツール引数")


class Tasks(BaseModel):
    tasks: list[Task] = Field(description="タスクのリスト")


class TaskResult(TypedDict):
    task: Task = Field(description="タスクの詳細")
    result_code: int = Field(description="タスクの実行結果コード")
    message: str = Field(description="タスクの実行結果メッセージ")


class ParallelRagAgentInputState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class ParallelRagAgentProcessingState(TypedDict):
    tasks: Tasks


class ParallelRagAgentOutputState(TypedDict):
    task_results: Annotated[list[TaskResult], operator.add]
    summary: str
    messages: Annotated[Sequence[BaseMessage], add_messages]


class ParallelRagAgentState(
    ParallelRagAgentInputState,
    ParallelRagAgentProcessingState,
    ParallelRagAgentOutputState,
):
    pass
