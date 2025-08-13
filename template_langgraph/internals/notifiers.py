"""Scraper interfaces and implementations for NewsSummarizerAgent.

This module defines an abstract base scraper so different scraping strategies
(mock, httpx-based, future headless browser, etc.) can be plugged into the agent
without changing orchestration logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class NotifierType(str, Enum):
    MOCK = "mock"
    SLACK = "slack"


class Settings(BaseSettings):
    notifier_type: NotifierType = NotifierType.MOCK
    notifier_slack_webhook_url: str = "https://hooks.slack.com/services/Txxx/Bxxx/xxx"

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


class SlackNotifier(BaseNotifier):
    """Slack notifier for sending notifications to a Slack channel."""

    def __init__(self, settings=get_notifier_settings()):
        self.webhook_url = settings.notifier_slack_webhook_url

    def notify(self, text: str):
        logger.info(f"Slack notify with text: {text}")
        with httpx.Client() as client:
            client.post(
                self.webhook_url,
                json={
                    "text": text,
                },
            )


def get_notifier(settings: Settings = None) -> BaseNotifier:
    if settings is None:
        settings = get_notifier_settings()

    if settings.notifier_type == NotifierType.MOCK:
        return MockNotifier()
    elif settings.notifier_type == NotifierType.SLACK:
        return SlackNotifier(settings)
    else:
        raise ValueError(f"Unknown notifier type: {settings.notifier_type}")
