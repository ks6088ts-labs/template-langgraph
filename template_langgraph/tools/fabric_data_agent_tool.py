from functools import lru_cache
from uuid import uuid4

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from template_langgraph.loggers import get_logger

logger = get_logger(__name__)
logger.setLevel("DEBUG")


class Settings(BaseSettings):
    fabric_data_agent_name: str = "your_fabric_data_agent_name"
    fabric_data_agent_user_instructions: str = "your_fabric_data_agent_user_instructions"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_fabric_data_agent_settings() -> Settings:
    """Get Fabric Data Agent settings."""
    return Settings()


# FIXME: properly implement Fabric Data Agent client
class FabricDataAgentWrapper:
    def __init__(
        self,
        settings: Settings = None,
    ):
        if settings is None:
            settings = get_fabric_data_agent_settings()
        self.settings = settings

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Search for similar documents in Fabric Data Agent."""
        logger.info(f"Just returning mock results for query: {query}, k: {k}")
        return [
            {
                "content": f"mock result #1 for {query}",
                "id": str(uuid4()),
            },
            {
                "content": f"mock result #2 for {query}",
                "id": str(uuid4()),
            },
        ]


class FabricDataAgentInput(BaseModel):
    query: str = Field(
        default="禅モード",
        description="Query to search in the Fabric Data Agent",
    )
    k: int = Field(
        default=5,
        description="Number of results to return from the similarity search",
    )


class FabricDataAgentOutput(BaseModel):
    content: str = Field(description="Content of the document")
    id: str = Field(description="ID of the document")


@tool(args_schema=FabricDataAgentInput)
def search_fabric_data_agent(query: str, k: int = 5) -> list[FabricDataAgentOutput]:
    """Search for similar documents in Fabric Data Agent.

    Args:
        query: The search query string
        k: Number of results to return (default: 5)

    Returns:
        FabricDataAgentOutput: A Pydantic model containing the search results
    """
    # FIXME: do not create every time
    wrapper = FabricDataAgentWrapper()
    results = wrapper.search(query=query, k=k)
    outputs = []
    for result in results:
        outputs.append(
            FabricDataAgentOutput(
                content=result["content"],
                id=result["id"],
            ),
        )
    return outputs
