from pydantic import BaseModel, Field
from typing import List, Dict, Any


class ClassifyRequest(BaseModel):
    """Схема запроса для классификации текста."""
    text: str = Field(
        ...,
        description="Текст для классификации",
        min_length=1,
        max_length=5000,
        examples=["Этот сервис работает отлично!"]
    )


class PredictionResult(BaseModel):
    """Схема одного предсказания."""
    label: str = Field(..., description="Класс/метка")
    score: float = Field(..., description="Уверенность модели от 0 до 1")


class ClassifyResponse(BaseModel):
    """Схема ответа API."""
    text: str = Field(..., description="Исходный текст")
    predictions: List[PredictionResult] = Field(
        ..., 
        description="Список предсказаний"
    )
    model_name: str = Field(..., description="Имя использованной модели")


class HealthResponse(BaseModel):
    """Схема ответа для health-check."""
    status: str
    model_loaded: bool
    model_name: str