"""Summarizer interfaces and implementations for NewsSummarizerAgent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from template_langgraph.agents.news_summarizer_agent.models import StructuredArticle
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


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


__all__ = [
    "BaseSummarizer",
    "MockSummarizer",
    "LlmSummarizer",
]
