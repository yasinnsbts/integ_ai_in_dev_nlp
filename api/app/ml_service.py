import logging
from typing import List, Dict, Any
from transformers import pipeline

logger = logging.getLogger(__name__)


class NLPClassifier:
    """
    Сервис для загрузки и использования ML-модели.
    
    Модель загружается ОДИН РАЗ при старте приложения (через lifespan),
    а не при каждом запросе — это критично для производительности.
    """
    
    def __init__(self):
        self.model = None
        # ЗАМЕТКА: Замените это на путь к вашей модели из репозитория
        # Например: "./data/saved_model" или "yasinnsbts/my-custom-model"
        self.model_name: str = "cointegrated/rubert-tiny-sentiment-balanced"
    
    def load_model(self) -> None:
        """Загрузка модели в память. Вызывается при старте приложения."""
        logger.info(f"🔄 Начало загрузки модели: {self.model_name}")
        try:
            # pipeline автоматически загружает модель и токенизатор
            self.model = pipeline(
                task="text-classification",
                model=self.model_name,
                device=-1  # -1 = CPU. Используйте 0 для GPU
            )
            logger.info(f"✅ Модель '{self.model_name}' успешно загружена!")
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке модели: {e}")
            raise
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Выполняет инференс модели на входном тексте.
        
        Args:
            text: Текст для классификации
            
        Returns:
            Список предсказаний вида [{"label": "...", "score": 0.95}, ...]
        """
        if self.model is None:
            raise RuntimeError(
                "Модель не загружена. Вызовите load_model() при старте приложения."
            )
        
        # pipeline возвращает список словарей
        result = self.model(text)
        
        # Если pipeline возвращает вложенный список (для batch), разворачиваем
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            result = result[0]
        
        return result
    
    def is_loaded(self) -> bool:
        """Проверяет, загружена ли модель."""
        return self.model is not None