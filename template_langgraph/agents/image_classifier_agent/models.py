import operator
from typing import Annotated

from pydantic import BaseModel, Field


class ClassifyImageState(BaseModel):
    prompt: str = Field(..., description="Prompt for classification")
    file_path: str = Field(..., description="Image file path")


class Result(BaseModel):
    title: str = Field(..., description="Title of the image")
    summary: str = Field(..., description="Summary of the image")
    labels: list[str] = Field(..., description="Labels extracted from the image")
    reliability: float = Field(..., description="Reliability score of the classification from 0 to 1")


class Results(BaseModel):
    file_path: str = Field(..., description="Image file path")
    result: Result = Field(..., description="Structured representation of the image classification result")


class AgentInputState(BaseModel):
    prompt: str = Field(..., description="Prompt for the agent")
    id: str = Field(..., description="Unique identifier for the request")
    file_paths: list[str] = Field(..., description="List of image file paths")


class AgentState(BaseModel):
    input: AgentInputState = Field(..., description="Input state for the agent")
    results: Annotated[list[Results], operator.add]
