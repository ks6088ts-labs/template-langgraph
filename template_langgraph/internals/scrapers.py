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
from youtube_transcript_api import YouTubeTranscriptApi

from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class ScraperType(str, Enum):
    MOCK = "mock"
    HTTPX = "httpx"
    YOUTUBE_TRANSCRIPT = "youtube_transcript"


class Settings(BaseSettings):
    scraper_type: ScraperType = ScraperType.MOCK

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_scraper_settings() -> Settings:
    """Get scraper settings."""
    return Settings()


class BaseScraper(ABC):
    """Abstract base scraper.

    Implementations should raise exceptions outward only when they are truly
    unrecoverable; transient network issues should surface as httpx.RequestError
    so the caller can decide how to handle them.
    """

    @abstractmethod
    def scrape(self, url: str) -> str:  # pragma: no cover - interface
        """Retrieve raw textual/HTML content for the given URL.

        Args:
            url: Target URL.
        Returns:
            str: Raw content (HTML / text).
        """
        raise NotImplementedError


class MockScraper(BaseScraper):
    """Deterministic scraper for tests / offline development."""

    def scrape(self, url: str) -> str:
        logger.info(f"Mock scrape for URL: {url}")
        return "<html><body><h1>Mocked web content</h1></body></html>"


class HttpxScraper(BaseScraper):
    """Simple httpx based scraper."""

    def scrape(self, url: str) -> str:
        logger.info(f"Fetching URL via httpx: {url}")
        with httpx.Client() as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text


class YouTubeTranscriptScraper(BaseScraper):
    """YouTube transcript scraper."""

    def scrape(self, url: str) -> str:
        logger.info(f"Fetching YouTube transcript for URL: {url}")
        video_id = url.split("v=")[-1].split("&")[0]
        transcript = YouTubeTranscriptApi().fetch(
            video_id=video_id,
            languages=["ja", "en"],
        )
        text_list = [item.text for item in transcript]
        return " ".join(text_list)


def get_scraper(settings: Settings = None) -> BaseScraper:
    if settings is None:
        settings = get_scraper_settings()

    if settings.scraper_type == ScraperType.MOCK:
        return MockScraper()
    elif settings.scraper_type == ScraperType.HTTPX:
        return HttpxScraper()
    elif settings.scraper_type == ScraperType.YOUTUBE_TRANSCRIPT:
        return YouTubeTranscriptScraper()
    else:
        raise ValueError(f"Unknown scraper type: {settings.scraper_type}")
