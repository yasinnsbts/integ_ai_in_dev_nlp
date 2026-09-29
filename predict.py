"""Пример использования сохранённой модели: python predict.py \"Не могу войти в банк\"."""

import argparse
import json
from pathlib import Path

import joblib


def main():
    parser = argparse.ArgumentParser(description="Определить категорию обращения в банк")
    parser.add_argument("text", help="Текст обращения в кавычках")
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "models" / "bank_classifier.joblib")
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("Текст обращения не должен быть пустым.")
    if not args.model.exists():
        parser.error("Модель не найдена. Сначала выполните python train_model.py.")
    # Загружайте только свои/доверенные joblib-файлы.
    model = joblib.load(args.model)
    category = str(model.predict([args.text])[0])
    print(json.dumps({"text": args.text, "category": category}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
