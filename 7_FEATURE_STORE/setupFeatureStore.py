# 7_FEATURE_STORE/setup_feature_store.py
from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import json

ROOT = Path(__file__).resolve().parent
DB   = ROOT.parent / "6_DATA_TRANSFORMATION_AND_STORAGE" / "feature_store.db"
OUT  = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS fs_feature (
  name TEXT PRIMARY KEY,
  dtype TEXT NOT NULL,
  description TEXT,
  source TEXT,
  owner TEXT,
  tags TEXT,
  created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fs_feature_group (
  group_name TEXT NOT NULL,
  version INTEGER NOT NULL,
  primary_key TEXT NOT NULL,
  label TEXT,
  description TEXT,
  created_at_utc TEXT NOT NULL,
  PRIMARY KEY (group_name, version)
);

CREATE TABLE IF NOT EXISTS fs_feature_group_feature (
  group_name TEXT NOT NULL,
  version INTEGER NOT NULL,
  feature_name TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  PRIMARY KEY (group_name, version, feature_name),
  FOREIGN KEY (feature_name) REFERENCES fs_feature(name)
);
"""

def ts_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def table_columns(cur, table: str) -> list[tuple[str, str]]:
    cur.execute(f"PRAGMA table_info({table});")
    return [(r[1], (r[2] or "").upper()) for r in cur.fetchall()]  # (name, type)

def norm_dtype(sqlite_dtype: str) -> str:
    d = (sqlite_dtype or "").upper()
    if "INT" in d: return "INTEGER"
    if any(x in d for x in ["REAL","FLOA","DOUB","NUM"]): return "REAL"
    return "TEXT"

def upsert_feature(cur, name, dtype, description="", source="etl_step_6", owner="team", tags=None):
    cur.execute("""
      INSERT INTO fs_feature(name, dtype, description, source, owner, tags, created_at_utc)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(name) DO UPDATE SET
        dtype=excluded.dtype,
        description=excluded.description,
        source=excluded.source,
        owner=excluded.owner,
        tags=excluded.tags
    """, (name, dtype, description, source, owner, json.dumps(tags or []), ts_utc()))

def create_group(cur, group_name, version, primary_key, label=None, description=""):
    cur.execute("""
      INSERT OR REPLACE INTO fs_feature_group(group_name, version, primary_key, label, description, created_at_utc)
      VALUES (?,?,?,?,?,?)
    """, (group_name, version, primary_key, label, description, ts_utc()))

def set_group_features(cur, group_name, version, feature_names):
    cur.execute("DELETE FROM fs_feature_group_feature WHERE group_name=? AND version=?",
                (group_name, version))
    for i, f in enumerate(feature_names):
        cur.execute("""
          INSERT INTO fs_feature_group_feature(group_name, version, feature_name, ordinal)
          VALUES (?,?,?,?)
        """, (group_name, version, f, i))

def main():
    if not DB.exists():
        raise SystemExit(f"DB not found: {DB}")

    con = sqlite3.connect(DB)
    cur = con.cursor()

    # 1) Registry tables
    cur.executescript(REGISTRY_DDL)

    # 2) Inspect base feature table
    cols = table_columns(cur, "feature_store")
    if not cols:
        raise SystemExit("Table 'feature_store' is empty or missing.")
    col_names = [c for c,_ in cols]

    # 3) Identify primary key / label if present
    id_col     = "customer_id" if "customer_id" in col_names else col_names[0]
    label_col  = "churned"     if "churned" in col_names     else None

    # 4) Register every column as a feature
    for name, t in cols:
        dtype = norm_dtype(t)
        tags = []
        if name == id_col:    tags.append("entity_key")
        if name == label_col: tags.append("label")
        if name.startswith(("platform_","location_","subscription_")): tags.append("one_hot")
        upsert_feature(cur, name, dtype,
                       description=f"Auto-registered from feature_store.{name}",
                       tags=tags)

    # 5) Build groups that include ALL columns (ID + label first)
    base_order = []
    if id_col in col_names:    base_order.append(id_col)
    if label_col in col_names: base_order.append(label_col)
    rest = [c for c in col_names if c not in base_order]
    feature_list_all = base_order + rest

    create_group(cur, "churn_all", 1, primary_key=id_col, label=label_col,
                 description="All columns from feature_store")
    set_group_features(cur, "churn_all", 1, feature_list_all)

    create_group(cur, "churn_v1", 1, primary_key=id_col, label=label_col,
                 description="All columns (default group identical to churn_all v1)")
    set_group_features(cur, "churn_v1", 1, feature_list_all)

    con.commit()
    con.close()

    manifest = {
        "db": str(DB),
        "groups_created": [
            {"name": "churn_all", "version": 1, "features": len(feature_list_all)},
            {"name": "churn_v1",  "version": 1, "features": len(feature_list_all)}
        ],
        "primary_key": id_col,
        "label": label_col,
        "created_at_utc": ts_utc()
    }
    (OUT / "feature_registry_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[OK] Feature registry initialized and all-features groups created.")

if __name__ == "__main__":
    main()
