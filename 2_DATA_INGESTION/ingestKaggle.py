# 2_DATA_INGESTION/ingestKaggle.py
import os
import json
import zipfile
import glob
import shutil
import time
from datetime import datetime, timezone

import pandas as pd


# ---------- helpers ----------
def ts_utc() -> str:
    """ISO-8601 UTC with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# supports ~/.kaggle/kaggle.json or local keys.txt
def _load_kaggle_env():
    """Load KAGGLE_USERNAME / KAGGLE_KEY from keys.txt if present."""
    keys_path = os.path.join(os.path.dirname(__file__), "keys.txt")
    if os.path.exists(keys_path):
        with open(keys_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()


# ---------- main download ----------
def download_from_kaggle():
    """
    Downloads the streaming subscription churn dataset, extracts into a temp folder,
    moves exactly one CSV to staging/kaggle/StreamingCustomerChurn_KAGGLE.csv,
    and cleans up everything else so only that one CSV remains.
    """
    _load_kaggle_env()
    from kaggle.api.kaggle_api_extended import KaggleApi

    dataset = "raghunandan9605/streaming-subscription-churn-dataset"

    base_dir = os.path.dirname(__file__)
    staging_dir = os.path.join(base_dir, "staging", "kaggle")
    os.makedirs(staging_dir, exist_ok=True)

    # temp extract dir so we can cleanly remove everything except the final file
    tmp_dir = os.path.join(staging_dir, f"_tmp_{int(time.time())}")
    os.makedirs(tmp_dir, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    # Download zip(s) into tmp, then extract there
    api.dataset_download_files(dataset, path=tmp_dir, unzip=False)

    zip_files = glob.glob(os.path.join(tmp_dir, "*.zip"))
    for z in zip_files:
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(tmp_dir)
        # remove zip after extraction
        try:
            os.remove(z)
        except OSError:
            pass

    # find CSVs inside tmp only
    csv_paths = glob.glob(os.path.join(tmp_dir, "**", "*.csv"), recursive=True)
    if not csv_paths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise FileNotFoundError("No CSV found after unzip.")

    # pick the largest CSV as before
    original_csv = max(csv_paths, key=lambda p: os.path.getsize(p))

    # final destination directly under staging/kaggle
    final_csv = os.path.join(staging_dir, "StreamingCustomerChurn_KAGGLE.csv")

    # Ensure destination dir exists
    os.makedirs(os.path.dirname(final_csv), exist_ok=True)

    # Replace if it already exists (atomic on Windows too)
    os.replace(original_csv, final_csv)

    # Clean up everything else so only final file remains
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Small retry to handle transient file locks (AV/indexer)
    df = None
    for _ in range(5):
        try:
            df = pd.read_csv(final_csv)
            break
        except PermissionError:
            time.sleep(0.5)
    if df is None:
        # last attempt raises if still locked
        df = pd.read_csv(final_csv)

    return {
        "source": "kaggle",
        "dataset": dataset,
        "file": os.path.relpath(final_csv, base_dir),
        "rows": int(len(df)),
    }


def write_manifest(status, results):
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "timestamp": ts_utc(),
        "status": status,
        "runs": results,
    }
    with open(os.path.join(out_dir, "manifestKaggle.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def append_log(status, results):
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "ingestion.log"), "a", encoding="utf-8") as f:
        f.write(f"{ts_utc()} - {status} - {results}\n")


# ---------- entrypoint ----------
if __name__ == "__main__":
    results = []
    try:
        res = download_from_kaggle()
        results.append(res)
        status = "success"
    except Exception as e:
        results.append({"error": str(e)})
        status = "failed"

    write_manifest(status, results)
    append_log(status, results)
    print(status, results)
