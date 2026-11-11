FROM python:3.12-slim
RUN pip install --no-cache-dir "mlflow==3.18.0" "scikit-learn==1.9.1"
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--artifacts-destination", "/mlflow-data/artifacts", \
     "--serve-artifacts", \
     "--allowed-hosts", "*"]
