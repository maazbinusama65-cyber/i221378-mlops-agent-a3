FROM apache/airflow:3.1.0
# Training dependencies for the DAG tasks (client-only MLflow: the server runs in its own container).
RUN pip install --no-cache-dir "mlflow-skinny==3.18.0" "scikit-learn==1.9.1" "pyyaml"
ENV PYTHONPATH=/opt/airflow/project
