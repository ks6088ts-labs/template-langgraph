from functools import lru_cache

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.documents import Document
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from template_langgraph.llms.azure_openais import AzureOpenAiWrapper


class Settings(BaseSettings):
    ai_search_key: str = "<your-ai-search-key>"
    ai_search_endpoint: str = "<your-ai-search-endpoint>"
    ai_search_index_name: str = "<your-ai-index-name>"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_ai_search_settings() -> Settings:
    """Get AI Search settings."""
    return Settings()


class AiSearchClientWrapper:
    def __init__(
        self,
        settings: Settings = None,
    ):
        if settings is None:
            settings = get_ai_search_settings()
        self.vector_store: AzureSearch = AzureSearch(
            azure_search_endpoint=settings.ai_search_endpoint,
            azure_search_key=settings.ai_search_key,
            index_name=settings.ai_search_index_name,
            embedding_function=AzureOpenAiWrapper().embedding_model.embed_query,
        )

    def add_documents(
        self,
        documents: list[Document],
    ) -> list[str]:
        """Add documents to a Cosmos DB container."""
        return self.vector_store.add_documents(
            documents=documents,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 5,
    ) -> list[Document]:
        """Perform a similarity search in the Cosmos DB index."""
        return self.vector_store.similarity_search(
            query=query,
            k=k,  # Number of results to return
        )


class AiSearchInput(BaseModel):
    query: str = Field(
        default="禅モード",
        description="Query to search in the AI Search index",
    )
    k: int = Field(
        default=5,
        description="Number of results to return from the similarity search",
    )


class AiSearchOutput(BaseModel):
    content: str = Field(description="Content of the document")
    id: str = Field(description="ID of the document")


@tool(args_schema=AiSearchInput)
def search_ai_search(query: str, k: int = 5) -> list[AiSearchOutput]:
    """Search for similar documents in AI Search index.

    Args:
        query: The search query string
        k: Number of results to return (default: 5)

    Returns:
        AiSearchOutput: A Pydantic model containing the search results
    """
    wrapper = AiSearchClientWrapper()
    documents = wrapper.similarity_search(
        query=query,
        k=k,
    )
    outputs = []
    for document in documents:
        outputs.append(
            {
                "content": document.page_content,
                "id": document.id,
            }
        )
    return outputs
