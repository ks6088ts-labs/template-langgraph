import logging
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ObjectTag(Enum):
    Book = "Book"
    Person = "Person"
    Car = "Car"
    Dog = "Dog"
    Cat = "Cat"
    # Add more tags as needed


# https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/concept-object-detection-40
@dataclass
class BoundingBox:
    left: int
    top: int
    width: int
    height: int


class ObjectDetectionResponse(BaseModel):
    """
    Object Detection Response Model
    """

    name: ObjectTag = Field(description="Detected object tag")
    confidence: float = Field(description="Confidence score of the detection (0 to 1)")
    bounding_box: BoundingBox = Field(description="Bounding box of the detected object")

    def __str__(self) -> str:
        return self.model_dump_json(indent=2)


class ObjectDetectionResult(BaseModel):
    """
    Object Detection Result Model
    """

    objects: list[ObjectDetectionResponse] = Field(description="List of detected objects")

    def __str__(self) -> str:
        return self.model_dump_json(indent=2)


class ImageCaptioningResult(BaseModel):
    """
    Image Captioning Result Model
    """

    caption: str = Field(description="Caption of the image")
    confidence: float = Field(description="Confidence score of the caption")

    def __str__(self) -> str:
        return self.model_dump_json(indent=2)
