# 10_Orchestrate/dmmlPipeline.py
from __future__ import annotations
import os, sys, subprocess
from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

# Repo root is the parent of 10_Orchestrate/
REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable  # use the worker's current interpreter

def run_script(rel_path: str, *args: str) -> None:
    """Run a repo script with the repo on PYTHONPATH, from its own directory."""
    script = REPO_ROOT / rel_path
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    subprocess.run(
        [PYTHON, str(script), *args],
        cwd=str(script.parent),
        env=env,
        check=True,
    )

default_args = {
    "owner": "dmml",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
        dag_id="dmml_pipeline",
        description="DMML E2E pipeline (ingest → raw store → validate → prep → FS → version → train)",
        default_args=default_args,
        start_date=datetime(2025, 1, 1),
        schedule_interval=None,   # trigger manually; set '@daily' if you want a schedule
        catchup=False,
        max_active_runs=1,
        tags=["dmml", "assignment", "e2e"],
) as dag:

    # -------------------- Ingestion (parallel) --------------------
    with TaskGroup(group_id="ingestion") as ingestion:
        ingest_from_csv = PythonOperator(
            task_id="ingest_from_csv",
            python_callable=run_script,
            op_args=["2_DATA_INGESTION/ingestFromCsv.py"],
        )

        ingest_kaggle = PythonOperator(
            task_id="ingest_kaggle",
            python_callable=run_script,
            op_args=["2_DATA_INGESTION/ingestKaggle.py"],
        )

    # -------------------- Raw storage → Validation → Preparation --------------------
    store_raw = PythonOperator(
        task_id="store_data_lake",
        python_callable=run_script,
        op_args=["3_RAW_DATA_STORAGE/storeDataLake.py"],
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=run_script,
        op_args=["4_DATA_VALIDATION/validateData.py"],
    )

    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=run_script,
        op_args=["5_DATA_PREPARATION/prepareData.py"],
    )

    # -------------------- Feature store load --------------------
    load_feature_store = PythonOperator(
        task_id="load_feature_store",
        python_callable=run_script,
        op_args=["6_DATA_TRANSFORMATION_AND_STORAGE/storeFeatureStore.py"],
    )

    # -------------------- Feature store utilities --------------------
    with TaskGroup(group_id="feature_store") as feature_store:
        setup_fs = PythonOperator(
            task_id="setup_feature_store",
            python_callable=run_script,
            op_args=["7_FEATURE_STORE/setupFeatureStore.py"],
        )

        export_docs = PythonOperator(
            task_id="export_feature_docs",
            python_callable=run_script,
            op_args=["7_FEATURE_STORE/export_feature_docs.py"],
        )

        get_training = PythonOperator(
            task_id="get_training_set",
            python_callable=run_script,
            op_args=["7_FEATURE_STORE/getTrainingSet.py"],
        )

        api_smoke = PythonOperator(
            task_id="feature_api_smoke",
            python_callable=run_script,
            op_args=["7_FEATURE_STORE/featureApi.py"],
        )

        # setup -> (export, get_training) in parallel -> api smoke
        setup_fs >> [export_docs, get_training] >> api_smoke

    # -------------------- Data versioning (Git/LFS push) --------------------
    version_and_push = PythonOperator(
        task_id="version_and_push",
        python_callable=run_script,
        op_args=["8_DATA_VERSIONING/uploadToGithub.py"],
    )

    # -------------------- Model training --------------------
    train_models = PythonOperator(
        task_id="train_models",
        python_callable=run_script,
        op_args=["9_MODEL_BUILDING/trainModels.py"],  # matches your file name
    )

    # -------------------- Dependencies --------------------
    ingestion >> store_raw >> validate >> prepare >> load_feature_store
    load_feature_store >> feature_store >> version_and_push >> train_models
