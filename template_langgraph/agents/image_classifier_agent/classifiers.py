"""Classifier interfaces and implementations for ImageClassifierAgent.

This module defines an abstract base classifier interface so that different
image classification strategies (mock, LLM-backed, future vision models, etc.)
can be plugged into the agent without modifying the agent orchestration code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from template_langgraph.agents.image_classifier_agent.models import Result
from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class BaseClassifier(ABC):
    """Abstract base class for image classifiers.

    Implementations should return a structured ``Result`` object.
    The ``llm`` argument is kept generic (Any) to avoid tight coupling
    with a specific provider wrapper; callers supply a model instance
    that offers the needed interface (e.g. ``with_structured_output``).
    """

    @abstractmethod
    def predict(self, prompt: str, image: str, llm: BaseChatModel) -> Result:  # pragma: no cover - interface
        """Classify an image.

        Args:
            prompt: Instruction or question guiding the classification.
            image: Base64-encoded image string ("data" portion only).
            llm: A language / vision model instance used (if needed) by the classifier.

        Returns:
            Result: Structured classification output.
        """
        raise NotImplementedError


class MockClassifier(BaseClassifier):
    """Simple mock classifier used for tests / offline development."""

    def predict(self, prompt: str, image: str, llm: Any) -> Result:  # noqa: D401
        import time

        time.sleep(3)  # Simulate a long-running process
        return Result(
            title="Mocked Image Title",
            summary=f"Mocked summary of the prompt: {prompt}",
            labels=["mocked_label_1", "mocked_label_2"],
            reliability=0.95,
        )


class LlmClassifier(BaseClassifier):
    """LLM-backed classifier using the provided model's structured output capability."""

    def predict(self, prompt: str, image: str, llm: BaseChatModel) -> Result:  # noqa: D401
        logger.info(f"Classifying image with LLM: {prompt}")
        return llm.with_structured_output(Result).invoke(
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source_type": "base64",
                            "data": image,
                            "mime_type": "image/png",
                        },
                    ],
                }
            ]
        )
