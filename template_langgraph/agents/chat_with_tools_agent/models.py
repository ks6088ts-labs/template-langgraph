from collections.abc import Sequence
from typing import (
    Annotated,
    TypedDict,
)

from langchain_core.messages import (
    BaseMessage,
)
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
