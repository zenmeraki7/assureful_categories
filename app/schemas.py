"""
app/schemas.py

Defines all Pydantic models for API request validation and response
serialization. This ensures all API I/O is typed and validated.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# --- Input Schemas ---

class ProductInput(BaseModel):
    """
    Schema for a single product to be classified.
    Inputs are sanitized by Pydantic.
    """
    title: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The product title."
    )
    description: Optional[str] = Field(None, max_length=5000)
    tags: Optional[str] = Field(None, max_length=1000)
    product_type: Optional[str] = Field(None, max_length=100)
    vendor: Optional[str] = Field(None, max_length=100)

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Men's Classic Cotton T-Shirt",
                "description": "A comfortable 100% cotton t-shirt.",
                "vendor": "ClassicWear"
            }
        }


class FeedbackInput(BaseModel):
    """
    Schema for submitting manual feedback.
    """
    product_input: ProductInput = Field(
        ...,
        description="The original product data that was classified."
    )
    predicted_path: str = Field(
        ...,
        description="The incorrect category path the model predicted."
    )
    correct_path: str = Field(
        ...,
        description="The ground-truth category path from the user."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "product_input": {
                    "title": "Men's Classic Cotton T-Shirt",
                    "vendor": "ClassicWear"
                },
                "predicted_path": "Apparel/Men/Shirts",
                "correct_path": "Apparel/Men/T-Shirts"
            }
        }

# --- Output Schemas ---

class LevelResult(BaseModel):
    """A single level in the predicted category hierarchy."""
    name: str
    path: str
    similarity: float
    confidence: str

class AlternativeResult(BaseModel):
    """A single alternative (lower-confidence) prediction."""
    category_id: str
    category_path: str
    similarity: float
    confidence: str
    depth: int

class ClassificationResponse(BaseModel):
    """
    The main response from the /classify endpoint.
    All fields are automatically type-checked and formatted.
    """
    product_title: str
    category_id: str
    category_path: str
    similarity: float
    confidence: str
    depth: int
    max_depth: int
    prediction_method: str
    levels: Dict[str, LevelResult]
    alternatives: List[AlternativeResult] = Field(default_factory=list)