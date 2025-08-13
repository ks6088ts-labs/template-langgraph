import os
from functools import lru_cache
from glob import glob

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    csv_loader_data_dir_path: str = "./data"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_csv_loader_settings() -> Settings:
    """Get CSV loader settings."""
    return Settings()


class CsvLoaderWrapper:
    def __init__(
        self,
        settings: Settings = None,
    ):
        if settings is None:
            settings = get_csv_loader_settings()
        self.settings = settings

    def load_csv_docs(self) -> list[Document]:
        """Load CSV documents from the specified directory."""
        csv_path = glob(
            os.path.join(self.settings.csv_loader_data_dir_path, "**", "*.csv"),
            recursive=True,
        )
        docs = []

        for path in csv_path:
            loader = CSVLoader(file_path=path)
            docs.extend(loader.load())

        return docs
