import operator
from typing import (
    Annotated,
    TypedDict,
)

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
    query: str


class ParallelRagAgentProcessingState(TypedDict):
    tasks: Tasks


class ParallelRagAgentOutputState(TypedDict):
    task_results: Annotated[list[TaskResult], operator.add]
    summary: str


class ParallelRagAgentState(
    ParallelRagAgentInputState,
    ParallelRagAgentProcessingState,
    ParallelRagAgentOutputState,
):
    pass
