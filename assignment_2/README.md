# Loan Default Prediction Pipeline

Medallion-architecture (bronze/silver/gold) data pipeline, orchestrated with Airflow, that builds
monthly feature and label stores for loan customers and trains, scores, and monitors a champion
(logistic regression) vs. challenger (XGBoost) loan default model.

## Structure
- `dags/` - Airflow DAG (`loan_default_pipeline`) wiring the label store, feature store, training,
  inference, and monitoring tasks together on a monthly schedule
- `utils/` - pipeline code: bronze/silver/gold processing, model training, inference, monitoring,
  and the CLI entry points the DAG invokes
- `data/` - raw source CSVs (loan, attributes, financials, clickstream)
- `datamart/` - bronze/silver/gold outputs written by the pipeline
- `model_bank/` - trained model artefacts and metadata
- `main.py` - standalone, non-Airflow entry point for a full local backfill + training run

## Running it
```
docker-compose up --build
```
Airflow webserver becomes available at `localhost:8080`; trigger the `loan_default_pipeline` DAG
from there.

## Github Link
https://github.com/zexellew2024/mle
