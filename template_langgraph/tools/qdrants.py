from functools import lru_cache

from langchain.tools import tool
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient
from qdrant_client.http.models import UpdateResult
from qdrant_client.models import Distance, PointStruct, VectorParams

from template_langgraph.llms.azure_openais import AzureOpenAiWrapper


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_qdrant_settings() -> Settings:
    """Get Qdrant settings."""
    return Settings()


class QdrantClientWrapper:
    def __init__(
        self,
        settings: Settings = None,
    ):
        if settings is None:
            settings = get_qdrant_settings()
        self.client = QdrantClient(
            url=settings.qdrant_url,
        )

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 1536,
    ) -> bool:
        """Create a collection in Qdrant."""
        result = self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        return result

    def delete_collection(
        self,
        collection_name: str,
    ) -> bool:
        """Delete a collection in Qdrant."""
        if self.client.collection_exists(collection_name=collection_name):
            self.client.delete_collection(collection_name=collection_name)
            return True
        return False

    def upsert_points(
        self,
        collection_name: str,
        points: list[PointStruct],
    ) -> UpdateResult:
        """Upsert points into a Qdrant collection."""
        return self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        limit: int = 3,
    ) -> list[PointStruct]:
        """Query points from a Qdrant collection."""
        return self.client.query_points(
            collection_name=collection_name,
            query=query,
            limit=limit,
        ).points


class QdrantInput(BaseModel):
    keywords: str = Field(description="Keywords to search")


class QdrantOutput(BaseModel):
    file_name: str = Field(description="The file name")
    content: str = Field(description="The content of the file")


@tool(args_schema=QdrantInput)
def search_qdrant(
    keywords: str,
) -> list[QdrantOutput]:
    """
    空想上のシステム「KABUTO」の過去のシステムのトラブルシュート事例が蓄積されたデータベースから、関連する情報を取得します。
    """
    wrapper = QdrantClientWrapper()
    query_vector = AzureOpenAiWrapper().create_embedding(keywords)
    results = wrapper.query_points(
        collection_name="qa_kabuto",
        query=query_vector,
        limit=3,
    )
    outputs = []
    for result in results:
        outputs.append(
            QdrantOutput(
                file_name=result.payload["file_name"],
                content=result.payload["content"],
            ),
        )
    return outputs
