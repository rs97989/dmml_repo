# 7_FEATURE_STORE/getOnlineVector.py
from pathlib import Path
import importlib.util
import sys
import sqlite3

HERE = Path(__file__).resolve().parent
API_PATH = HERE / "featureApi.py"

if not API_PATH.exists():
    raise SystemExit(f"feature_api.py not found at {API_PATH}")

spec = importlib.util.spec_from_file_location("feature_api", API_PATH)
feature_api = importlib.util.module_from_spec(spec)  # type: ignore
sys.modules["feature_api"] = feature_api
spec.loader.exec_module(feature_api)  # type: ignore

def pick_any_id():
    with feature_api.connect() as con:
        cur = con.cursor()
        try:
            cur.execute('SELECT "customer_id" FROM feature_store LIMIT 1')
            r = cur.fetchone()
            return r[0] if r else None
        except Exception:
            return None

def main():
    cid = pick_any_id()
    if cid is None:
        print("No rows found to sample.")
        return
    vec = feature_api.get_online_vector(cid, "churn_v1", 1)
    print("entity_id:", cid)
    print("feature_count:", len(vec))
    for k, v in list(vec.items())[:12]:
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
