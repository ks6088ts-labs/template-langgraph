"""Summarizer interfaces and implementations for NewsSummarizerAgent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from template_langgraph.agents.news_summarizer_agent.models import StructuredArticle
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class SummarizerType(str, Enum):
    """Enumeration of available summarizer types."""

    MOCK = "mock"
    LLM = "llm"


class Settings(BaseSettings):
    summarizer_type: SummarizerType = SummarizerType.MOCK

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_summarizer_settings() -> Settings:
    """Get summarizer settings."""
    return Settings()


class BaseSummarizer(ABC):
    """Abstract base summarizer returning a StructuredArticle."""

    @abstractmethod
    def summarize(self, prompt: str, content: str) -> StructuredArticle:  # pragma: no cover - interface
        """Summarize raw content using a given prompt."""
        raise NotImplementedError


class MockSummarizer(BaseSummarizer):
    """Deterministic summarizer for tests / offline development."""

    def summarize(self, prompt: str, content: str) -> StructuredArticle:  # noqa: D401
        return StructuredArticle(
            title="Mocked Title",
            date="2023-01-01",
            summary=f"Mocked summary of the content: {content}, prompt: {prompt}",
            keywords=["mock", "summary"],
            score=75,
        )


class LlmSummarizer(BaseSummarizer):
    """LLM backed summarizer leveraging structured output."""

    def __init__(self, llm: BaseChatModel | Any = AzureOpenAiWrapper().chat_model):
        self.llm = llm

    def summarize(self, prompt: str, content: str) -> StructuredArticle:  # noqa: D401
        logger.info(f"Summarizing input with LLM: {prompt}")
        return self.llm.with_structured_output(StructuredArticle).invoke(
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ]
        )


def get_summarizer(settings: Settings = None) -> BaseSummarizer:
    if settings is None:
        settings = get_summarizer_settings()

    if settings.summarizer_type == SummarizerType.MOCK:
        return MockSummarizer()
    elif settings.summarizer_type == SummarizerType.LLM:
        return LlmSummarizer()
    else:
        raise ValueError(f"Unknown summarizer type: {settings.summarizer_type}")
