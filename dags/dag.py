import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import ShortCircuitOperator

PROJECT_DIR = "/opt/airflow/project"
sys.path.insert(0, PROJECT_DIR)

from utils.constants import END_DATE_STR, LABEL_END_DATE_STR, MODEL_TRAIN_DATE_STR  # noqa: E402

# logistic_regression (champion, prioritised for explainability) vs xgboost (challenger)
CHAMPION_MODEL_FILE = f"loan_default_logistic_regression_{MODEL_TRAIN_DATE_STR.replace('-', '_')}.pkl"
CHALLENGER_MODEL_FILE = f"loan_default_xgboost_{MODEL_TRAIN_DATE_STR.replace('-', '_')}.pkl"
CHAMPION_MODEL_PATH = os.path.join(PROJECT_DIR, "model_bank", CHAMPION_MODEL_FILE)
CHALLENGER_MODEL_PATH = os.path.join(PROJECT_DIR, "model_bank", CHALLENGER_MODEL_FILE)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def bash_task(task_id, script, extra_args=""):
    return BashOperator(
        task_id=task_id,
        bash_command=(
            f"cd {PROJECT_DIR} && "
            f'PYTHONPATH={PROJECT_DIR} python3 utils/{script} --snapshotdate "{{{{ ds }}}}"{extra_args}'
        ),
    )


with DAG(
    "loan_default_pipeline",
    default_args=default_args,
    description="Loan default monthly datamart, training, inference & monitoring pipeline",
    schedule_interval="0 0 1 * *",
    start_date=datetime(2023, 1, 1),
    # past END_DATE_STR so the label store can mature labels for snapshots up to END_DATE_STR
    end_date=datetime.strptime(LABEL_END_DATE_STR, "%Y-%m-%d"),
    catchup=True,
) as dag:

    # --- label store ---
    dep_check_source_label_data = DummyOperator(task_id="dep_check_source_label_data")
    bronze_label_store = bash_task("run_bronze_label_store", "bronze_label_store.py")
    silver_label_store = bash_task("run_silver_label_store", "silver_label_store.py")
    gold_label_store = bash_task("run_gold_label_store", "gold_label_store.py")
    label_store_completed = DummyOperator(task_id="label_store_completed")

    dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed

    # --- feature store (source feature CSVs only cover up to END_DATE_STR) ---
    dep_check_source_feature_data = DummyOperator(task_id="dep_check_source_feature_data")
    is_feature_data_available = ShortCircuitOperator(
        task_id="is_feature_data_available",
        python_callable=lambda ds: ds <= END_DATE_STR,
        op_kwargs={"ds": "{{ ds }}"},
    )
    bronze_feature_store = bash_task("run_bronze_feature_store", "bronze_feature_store.py")
    silver_feature_store = bash_task("run_silver_feature_store", "silver_feature_store.py")
    gold_feature_store = bash_task("run_gold_feature_store", "gold_feature_store.py")
    feature_store_completed = DummyOperator(task_id="feature_store_completed")

    (dep_check_source_feature_data >> is_feature_data_available >> bronze_feature_store >>
     silver_feature_store >> gold_feature_store >> feature_store_completed)

    # --- model training (once, as of MODEL_TRAIN_DATE_STR) ---
    is_training_date = ShortCircuitOperator(
        task_id="is_training_date",
        python_callable=lambda ds: ds == MODEL_TRAIN_DATE_STR,
        op_kwargs={"ds": "{{ ds }}"},
    )
    model_training = bash_task("run_model_training", "run_model_training.py")
    model_comparison = bash_task(
        "model_comparison", "model_comparison.py",
        f" --champion {CHAMPION_MODEL_FILE} --challenger {CHALLENGER_MODEL_FILE}",
    )
    model_training_completed = DummyOperator(task_id="model_training_completed")

    [label_store_completed, feature_store_completed] >> is_training_date >> model_training >> model_comparison >> model_training_completed

    # checks the artefacts exist too, since inference has no dependency edge on model_training and
    # could otherwise start scoring before the model is saved on MODEL_TRAIN_DATE_STR itself
    is_monitoring_date = ShortCircuitOperator(
        task_id="is_monitoring_date",
        python_callable=lambda ds: (
            ds >= MODEL_TRAIN_DATE_STR
            and os.path.exists(CHAMPION_MODEL_PATH)
            and os.path.exists(CHALLENGER_MODEL_PATH)
        ),
        op_kwargs={"ds": "{{ ds }}"},
    )
    champion_inference = bash_task("champion_inference", "run_model_inference.py", f" --modelname {CHAMPION_MODEL_FILE}")
    challenger_inference = bash_task("challenger_inference", "run_model_inference.py", f" --modelname {CHALLENGER_MODEL_FILE}")
    model_inference_completed = DummyOperator(task_id="model_inference_completed")

    feature_store_completed >> is_monitoring_date
    is_monitoring_date >> champion_inference >> model_inference_completed
    is_monitoring_date >> challenger_inference >> model_inference_completed

    # --- model monitoring ---
    champion_monitor = bash_task("champion_monitor", "run_model_monitoring.py", f" --modelname {CHAMPION_MODEL_FILE}")
    challenger_monitor = bash_task("challenger_monitor", "run_model_monitoring.py", f" --modelname {CHALLENGER_MODEL_FILE}")
    model_monitor_completed = DummyOperator(task_id="model_monitor_completed")

    model_inference_completed >> champion_monitor >> model_monitor_completed
    model_inference_completed >> challenger_monitor >> model_monitor_completed

    # --- monitoring summary ---
    monitoring_summary = bash_task(
        "monitoring_summary", "model_monitoring_summary.py",
        f" --modelname {CHAMPION_MODEL_FILE} {CHALLENGER_MODEL_FILE}",
    )

    model_monitor_completed >> monitoring_summary
