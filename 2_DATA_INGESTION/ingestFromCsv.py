# 2_DATA_INGESTION/ingestFromCsv.py
import os
import json
import pandas as pd
from datetime import datetime, timezone

# ---------- helpers ----------
def _base_dir() -> str:
    return os.path.dirname(__file__)

def _ts_utc() -> str:
    # ISO 8601 with explicit UTC offset
    return datetime.now(timezone.utc).isoformat()

def _append_log(msg: str) -> None:
    base = _base_dir()
    log_dir = os.path.join(base, "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "ingestion.log"), "a", encoding="utf-8") as f:
        f.write(f"{_ts_utc()} - {msg}\n")

def _write_manifest(payload: dict) -> None:
    base = _base_dir()
    out_dir = os.path.join(base, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "manifestSource.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def _find_source_csv() -> str:
    """
    Find the intended CSV inside ./source.
    Prefers 'StreamingCustomerChurn_SOURCE.csv' but falls back to first *.csv it finds.
    """
    base = _base_dir()
    src_dir = os.path.join(base, "source")
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Source folder not found: {src_dir}")

    preferred = ["StreamingCustomerChurn_SOURCE.csv"]
    for name in preferred:
        candidate = os.path.join(src_dir, name)
        if os.path.exists(candidate):
            return candidate

    csvs = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.lower().endswith(".csv")]
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {src_dir}")
    return csvs[0]

# ---------- main ingestion ----------
def ingest_from_local_csv() -> dict:
    base = _base_dir()
    src_csv = _find_source_csv()

    # Read to validate + get row count; also normalizes line endings/encoding on save
    df = pd.read_csv(src_csv)

    staging_dir = os.path.join(base, "staging", "source")
    os.makedirs(staging_dir, exist_ok=True)
    dest_csv = os.path.join(staging_dir, "StreamingCustomerChurn_SOURCE.csv")
    df.to_csv(dest_csv, index=False)

    return {
        "source": "local_csv",
        "dataset": os.path.relpath(src_csv, base),
        "file": os.path.relpath(dest_csv, base),
        "rows": int(len(df)),
        "columns": df.shape[1],
    }

if __name__ == "__main__":
    try:
        run_info = ingest_from_local_csv()
        payload = {
            "timestamp": _ts_utc(),
            "status": "success",
            "run": run_info,
        }
        _write_manifest(payload)
        _append_log(f"success - {run_info}")
        print("success", run_info)
    except Exception as e:
        err = str(e)
        payload = {
            "timestamp": _ts_utc(),
            "status": "failed",
            "error": err,
        }
        _write_manifest(payload)
        _append_log(f"failed - {err}")
        print("failed", err)
