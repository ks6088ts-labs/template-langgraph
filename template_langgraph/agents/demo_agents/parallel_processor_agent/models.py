import operator
from typing import (
    Annotated,
    TypedDict,
)

from pydantic import BaseModel, Field


class Task(BaseModel):
    id: str = Field(description="タスクのID")
    description: str = Field(description="タスクの説明")
    state: str = Field(description="タスクの状態")


class Tasks(BaseModel):
    tasks: list[Task] = Field(description="タスクのリスト")


class TaskResult(TypedDict):
    task: Task = Field(description="タスクの詳細")
    result_code: int = Field(description="タスクの実行結果コード")
    message: str = Field(description="タスクの実行結果メッセージ")


class ParallelProcessorAgentInputState(TypedDict):
    goal: str


class ParallelProcessorAgentProcessingState(TypedDict):
    tasks: Tasks


class ParallelProcessorAgentOutputState(TypedDict):
    task_results: Annotated[list[TaskResult], operator.add]
    summary: str


class ParallelProcessorAgentState(
    ParallelProcessorAgentInputState,
    ParallelProcessorAgentProcessingState,
    ParallelProcessorAgentOutputState,
):
    pass
