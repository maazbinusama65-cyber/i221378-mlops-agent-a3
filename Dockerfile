# Churn inference service image (deployed to Kubernetes).
#   docker build -t churn-api:1.0.0 .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Pinned to the training environment so the pickled model loads identically.
RUN pip install --no-cache-dir "scikit-learn==1.9.1" "pandas>=2.1" "fastapi>=0.110" "uvicorn>=0.29" "joblib>=1.3"

COPY api/ ./api/
COPY models/model.joblib ./models/model.joblib

RUN useradd --uid 10001 --no-create-home app
USER 10001
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
