
FROM python:3.13-slim

WORKDIR /project

COPY api/requirements.txt /project/api/requirements.txt

RUN pip install --no-cache-dir -r /project/api/requirements.txt

COPY api/app/ /project/api/app/
COPY models/bank_classifier.joblib /project/models/bank_classifier.joblib

WORKDIR /project/api

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
