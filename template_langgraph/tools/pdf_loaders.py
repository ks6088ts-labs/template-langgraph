import os
from functools import lru_cache
from glob import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    pdf_loader_data_dir_path: str = "./data"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_pdf_loader_settings() -> Settings:
    """Get pdf loader settings."""
    return Settings()


class PdfLoaderWrapper:
    def __init__(
        self,
        settings: Settings = None,
    ):
        if settings is None:
            settings = get_pdf_loader_settings()
        self.settings = settings

    def load_pdf_docs(self) -> list[Document]:
        """Load pdf documents from the specified directory."""
        pdf_path = glob(
            os.path.join(self.settings.pdf_loader_data_dir_path, "**", "*.pdf"),
            recursive=True,
        )
        docs = []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        for path in pdf_path:
            loader = PyPDFLoader(path)
            pages = loader.load_and_split(text_splitter)
            docs.extend(pages)

        return docs
