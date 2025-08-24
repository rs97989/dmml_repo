# 3_RAW_DATA_STORAGE/storeDataLake.py
from pathlib import Path
from datetime import datetime, timezone
import shutil
import json

# ----- Paths -----
PROJECT_ROOT   = Path(__file__).resolve().parents[1]
INGEST_STAGING = PROJECT_ROOT / "2_DATA_INGESTION" / "staging"
RAW_ROOT       = Path(__file__).resolve().parent / "raw_data"
OUTPUTS        = Path(__file__).resolve().parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Target canonical filenames we expect to ingest
EXPECTED = {
    "kaggle": "StreamingCustomerChurn_KAGGLE.csv",
    "source": "StreamingCustomerChurn_SOURCE.csv",
}

# ----- Helpers -----
def ts_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def file_type_from_ext(p: Path) -> str:
    return (p.suffix or "").lower().lstrip(".") or "unknown"

def timestamp_folder() -> str:
    # keep folder-safe timestamp (no colons)
    return f"ingestion_ts={datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}"

def find_staged_file(source: str) -> Path:
    """
    Prefer the expected filename; otherwise fall back to
    the most recently modified CSV in the source's staging subfolder.
    """
    subdir = INGEST_STAGING / source
    expected = subdir / EXPECTED[source]
    if expected.exists():
        return expected

    # Fallback: latest CSV
    csvs = sorted(subdir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else expected  # return expected path (non-existent) to trigger "missing"

def copy_with_partitions(src: Path, source: str) -> dict:
    if not src.exists():
        return {"source": source, "type": "unknown", "status": "missing", "from": str(src)}

    ftype = file_type_from_ext(src)  # e.g., 'csv'
    dest_dir = RAW_ROOT / source / ftype / timestamp_folder()
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / src.name
    shutil.copy2(src, dest)

    return {
        "source": source,
        "type": ftype,
        "status": "copied",
        "from": str(src),
        "to": str(dest),
        "bytes": dest.stat().st_size,
    }

def write_manifest(results: list) -> dict:
    manifest = {"moved_at_utc": ts_utc(), "results": results}
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    with open(RAW_ROOT / "manifestRawStore.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest

def append_log(manifest: dict):
    """Append a human-readable line AND a JSON line to outputs/raw_store.log"""
    log_path = OUTPUTS / "raw_store.log"
    hr_parts = [manifest["moved_at_utc"]]
    for r in manifest["results"]:
        if r.get("status") == "copied":
            hr_parts.append(f"{r['source']}:{r['type']} COPIED ({r['bytes']} bytes)")
        elif r.get("status") == "missing":
            hr_parts.append(f"{r['source']} MISSING ({r['from']})")
        else:
            hr_parts.append(f"{r['source']} {r.get('status','unknown').upper()}")
    human_line = " | ".join(hr_parts)

    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(human_line + "\n")                  # human-readable
        lf.write(json.dumps(manifest) + "\n")        # JSON line

# ----- Main -----
def main():
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    kaggle_path = find_staged_file("kaggle")
    source_path = find_staged_file("source")

    files_to_move = [
        {"source": "kaggle", "path": kaggle_path},
        {"source": "source", "path": source_path},
    ]

    results = [copy_with_partitions(i["path"], i["source"]) for i in files_to_move]
    manifest = write_manifest(results)
    append_log(manifest)
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
