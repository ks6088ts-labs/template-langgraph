"""Scraper interfaces and implementations for NewsSummarizerAgent.

This module defines an abstract base scraper so different scraping strategies
(mock, httpx-based, future headless browser, etc.) can be plugged into the agent
without changing orchestration logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class NotifierType(str, Enum):
    MOCK = "mock"


class Settings(BaseSettings):
    notifier_type: NotifierType = NotifierType.MOCK

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_notifier_settings() -> Settings:
    """Get notifier settings."""
    return Settings()


class BaseNotifier(ABC):
    """Abstract base notifier."""

    @abstractmethod
    def notify(self, text: str):
        """Send a notification with the given text.

        Args:
            text: The text to include in the notification.

        """
        raise NotImplementedError


class MockNotifier(BaseNotifier):
    """Deterministic notifier for tests / offline development."""

    def notify(self, text: str):
        logger.info(f"Mock notify with text: {text}")


def get_notifier(settings: Settings = None) -> BaseNotifier:
    if settings is None:
        settings = get_notifier_settings()

    if settings.notifier_type == NotifierType.MOCK:
        return MockNotifier()
    else:
        raise ValueError(f"Unknown notifier type: {settings.notifier_type}")
