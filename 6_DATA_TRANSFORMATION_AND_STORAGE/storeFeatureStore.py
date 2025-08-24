# 6_DATA_TRANSFORMATION_AND_STORAGE/storeFeatureStore.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import json
import re
import pandas as pd
import numpy as np

# ---------------- Paths & constants ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_MATRIX = PROJECT_ROOT / "5_DATA_PREPARATION" / "outputs" / "model_matrix.csv"

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH    = OUT_DIR / "feature_store.db"
TABLE_NAME = "feature_store"

# logical names we want to preserve (case/spacing may differ in file)
LOGICAL_ID_COL     = "customer_id"
LOGICAL_TARGET_COL = "churned"
LOGICAL_SOURCE_COL = "__source__"

# SQLite typical parameter cap (safe lower than 999 to allow overhead)
SQLITE_PARAM_CAP = 980


# ---------------- Helpers ----------------
def ts_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_model_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"model_matrix not found: {path}")
    df = pd.read_csv(path)
    # strip whitespace from headers
    df.columns = [c.strip() for c in df.columns]
    return df


# --- name normalization & dedup (case-insensitive safe for SQLite) ---
_name_re = re.compile(r"[^a-z0-9_]+")

def normalize_name(name: str) -> str:
    """Lowercase, non-alnum -> '_', collapse repeats, strip, prefix if starts with digit."""
    s = name.strip().lower()
    s = _name_re.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if s == "":
        s = "col"
    if s[0].isdigit():
        s = f"c_{s}"
    return s


def sanitize_and_deduplicate_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Normalize and deduplicate columns in a case-insensitive way.
    Returns (df_with_unique_cols, rename_map: {original: new})
    """
    seen: dict[str, int] = {}   # key: normalized name, value: count
    new_cols: list[str] = []
    rename_map: dict[str, str] = {}

    for orig in df.columns:
        base = normalize_name(orig)
        key = base  # already lowercase
        if key not in seen:
            seen[key] = 0
            new_name = base
        else:
            seen[key] += 1
            new_name = f"{base}_{seen[key]}"

        new_cols.append(new_name)
        if new_name != orig:
            rename_map[orig] = new_name

    df2 = df.copy()
    df2.columns = new_cols
    return df2, rename_map


def resolve_special_names(rename_map: dict[str, str], cols: list[str]) -> tuple[str|None, str|None, str|None]:
    """
    After renaming, figure out the real column names for ID, TARGET, SOURCE.
    We match by normalized logical name against normalized new names or original->new map.
    """
    # Build reverse map: normalized original -> new
    rev = {normalize_name(k): v for k, v in rename_map.items()}
    norm_cols = {normalize_name(c): c for c in cols}

    def pick(logical: str) -> str | None:
        norm = normalize_name(logical)
        if norm in rev:
            return rev[norm]
        return norm_cols.get(norm, None)

    id_col     = pick(LOGICAL_ID_COL)
    target_col = pick(LOGICAL_TARGET_COL)
    source_col = pick(LOGICAL_SOURCE_COL)
    return id_col, target_col, source_col


def guess_sql_type(series: pd.Series) -> str:
    """Map pandas dtype to SQLite type suited for ML."""
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    return "TEXT"


def coerce_int_flags(df: pd.DataFrame, id_col: str|None, target_col: str|None, source_col: str|None) -> pd.DataFrame:
    """
    Cast one-hot columns and target to INTEGER (0/1). Ensure id is INTEGER if numeric, else TEXT.
    """
    df = df.copy()

    if target_col and target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).round().astype(int)

    for c in df.columns:
        if c in {id_col, target_col, source_col}:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        vals = s.fillna(0)
        # mark true one-hots as int flags
        if vals.isin([0, 1]).all():
            df[c] = vals.astype(int)

    if id_col and id_col in df.columns:
        try:
            df[id_col] = pd.to_numeric(df[id_col], errors="raise").astype(int)
        except Exception:
            df[id_col] = df[id_col].astype(str)

    return df


def build_create_table_sql(df: pd.DataFrame, id_col: str|None) -> str:
    parts = []
    for c in df.columns:
        sql_type = guess_sql_type(df[c])
        if id_col and c == id_col:
            if sql_type == "INTEGER":
                parts.append(f'"{c}" INTEGER PRIMARY KEY')
            else:
                parts.append(f'"{c}" TEXT PRIMARY KEY')
        else:
            parts.append(f'"{c}" {sql_type}')
    cols_sql = ",\n    ".join(parts)
    return f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} (\n    {cols_sql}\n);"


def create_table(conn: sqlite3.Connection, df: pd.DataFrame, id_col: str|None) -> None:
    sql = build_create_table_sql(df, id_col)
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")
    cur.execute(sql)
    conn.commit()


def _safe_chunksize(num_columns: int) -> int:
    """Compute max rows per multi-insert batch under SQLite's parameter cap."""
    # Each row contributes `num_columns` parameters; keep well below cap
    if num_columns <= 0:
        return 1000
    return max(1, SQLITE_PARAM_CAP // num_columns)


def insert_dataframe(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """
    Insert using a chunk size that respects SQLite's parameter limit.
    Falls back to row-by-row if needed (slow but safe).
    """
    cols = len(df.columns)
    chunksize = _safe_chunksize(cols)

    try:
        df.to_sql(
            TABLE_NAME,
            conn,
            if_exists="append",
            index=False,
            chunksize=chunksize,
            method="multi",  # multi-row insert with safe chunking
        )
    except sqlite3.OperationalError as e:
        # Fallback: single-row executemany (no 'multi'), avoids parameter cap issues
        print(f"[WARN] Falling back to single-row inserts due to: {e}")
        df.to_sql(
            TABLE_NAME,
            conn,
            if_exists="append",
            index=False,
            chunksize=1,
            method=None,
        )


def get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    return [row[1] for row in cur.fetchall()]


def sample_queries(conn: sqlite3.Connection, target_col: str|None, source_col: str|None) -> dict:
    cur = conn.cursor()
    out = {}

    cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    out["row_count"] = cur.fetchone()[0]

    cols = set(get_table_columns(conn, TABLE_NAME))
    if target_col and target_col in cols:
        cur.execute(f'''SELECT AVG("{target_col}") FROM {TABLE_NAME}''')
        out["churn_rate_overall"] = cur.fetchone()[0]

    if source_col and source_col in cols:
        cur.execute(f'''SELECT "{source_col}", COUNT(*) FROM {TABLE_NAME} GROUP BY "{source_col}"''')
        out["rows_by_source"] = cur.fetchall()

    return out


# ---------------- Main ----------------
def main():
    # 1) Load model matrix
    df = load_model_matrix(MODEL_MATRIX)

    # 2) Normalize + deduplicate column names (case-insensitive safe)
    df, rename_map = sanitize_and_deduplicate_columns(df)
    if rename_map:
        (OUT_DIR / "column_renames.json").write_text(json.dumps(rename_map, indent=2), encoding="utf-8")
        print(f"[INFO] Column names normalized/deduplicated. See {OUT_DIR / 'column_renames.json'}")

    # 3) Resolve special columns after rename
    id_col, target_col, source_col = resolve_special_names(rename_map, df.columns.tolist())

    # 4) Coerce types for ML-friendly storage
    df = coerce_int_flags(df, id_col, target_col, source_col)

    # 5) Reorder with keys first (if present)
    keys_first = [c for c in [id_col, target_col, source_col] if c and c in df.columns]
    ordered = keys_first + [c for c in df.columns if c not in keys_first]
    df = df[ordered]

    # 6) Create SQLite DB + table, then insert
    conn = sqlite3.connect(str(DB_PATH))
    try:
        # performance-friendly pragmas (optional)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=OFF;")
        create_table(conn, df, id_col)
        insert_dataframe(conn, df)
        stats = sample_queries(conn, target_col, source_col)
    finally:
        conn.commit()
        conn.close()

    # 7) Manifest
    manifest = {
        "stored_at_utc": ts_utc(),
        "model_matrix_csv": str(MODEL_MATRIX),
        "db_path": str(DB_PATH),
        "table": TABLE_NAME,
        "rows_loaded": int(len(df)),
        "columns_loaded": len(df.columns),
        "id_col": id_col,
        "target_col": target_col,
        "source_col": source_col,
        "renamed_columns": rename_map,  # {} if none
        "sample_stats": stats,
    }
    (OUT_DIR / "feature_store_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[OK] Feature store created.")
    print(f"DB: {DB_PATH}")
    print(f"Table: {TABLE_NAME}")
    print("ID/TARGET/SOURCE:", id_col, target_col, source_col)
    print("Sample stats:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
