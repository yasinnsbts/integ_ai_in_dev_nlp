from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from transformers import pipeline
import logging

# === НАСТРОЙКА ЛОГИРОВАНИЯ ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# === PYDANTIC СХЕМЫ ===
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class PredictionResult(BaseModel):
    label: str
    score: float

class ClassifyResponse(BaseModel):
    text: str
    predictions: List[PredictionResult]
    model_name: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str

# === ML СЕРВИС ===
class NLPClassifier:
    def __init__(self):
        self.model = None
        self.model_name = "cointegrated/rubert-tiny-sentiment-balanced"
    
    def load_model(self):
        logger.info(f"Загрузка модели: {self.model_name}")
        self.model = pipeline("text-classification", model=self.model_name)
        logger.info("Модель загружена!")
    
    def predict(self, text: str):
        if self.model is None:
            raise RuntimeError("Модель не загружена")
        return self.model(text)
    
    def is_loaded(self):
        return self.model is not None

ml_service = NLPClassifier()

# === LIFESPAN ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск приложения...")
    ml_service.load_model()
    yield
    logger.info("Остановка приложения...")
    ml_service.model = None

# === FASTAPI ПРИЛОЖЕНИЕ ===
app = FastAPI(
    title="NLP Classification API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/classify", response_model=ClassifyResponse)
async def classify_text(request: ClassifyRequest):
    try:
        predictions = ml_service.predict(request.text)
        return ClassifyResponse(
            text=request.text,
            predictions=predictions,
            model_name=ml_service.model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if ml_service.is_loaded() else "unhealthy",
        model_loaded=ml_service.is_loaded(),
        model_name=ml_service.model_name
    )

@app.get("/")
async def root():
    return {"message": "NLP API работает!", "docs": "/docs"}











