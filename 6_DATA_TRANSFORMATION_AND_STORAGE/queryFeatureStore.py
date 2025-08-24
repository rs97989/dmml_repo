# 6_DATA_TRANSFORMATION_AND_STORAGE/queryFeatureStore.py
from pathlib import Path
import sqlite3
import pandas as pd

DB_PATH  = Path(__file__).with_name("feature_store.db")
OUT_DIR  = Path(__file__).with_name("query_outputs")
OUT_DIR.mkdir(exist_ok=True)

def q(conn, name: str, sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run a SQL query, print a tiny summary, save to CSV, and return a DataFrame."""
    df = pd.read_sql_query(sql, conn, params=params or {})
    out = OUT_DIR / f"{name}.csv"
    df.to_csv(out, index=False)
    print(f"[{name}] rows={len(df)} -> {out}")
    return df

def list_columns(conn) -> list[str]:
    return pd.read_sql_query("PRAGMA table_info(feature_store);", conn)["name"].tolist()

def col_exists(conn, col: str) -> bool:
    return col in list_columns(conn)

def first_col_with_prefix(conn, prefix: str) -> str | None:
    for c in list_columns(conn):
        if c.startswith(prefix):
            return c
    return None

def main():
    if not DB_PATH.exists():
        raise SystemExit(f"DB not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    # --- schema dump ---
    q(conn, "schema", "PRAGMA table_info(feature_store);")

    # --- basic previews ---
    q(conn, "preview_10", "SELECT * FROM feature_store LIMIT 10;")
    q(conn, "row_count", "SELECT COUNT(*) AS rows FROM feature_store;")

    # --- churn metrics ---
    if col_exists(conn, "churned"):
        q(conn, "churn_rate_overall", "SELECT AVG(churned) AS churn_rate FROM feature_store;")

    if col_exists(conn, "source"):
        q(conn, "churn_by_source", """
            SELECT source, COUNT(*) AS n, AVG(churned) AS churn_rate
            FROM feature_store
            GROUP BY source
            ORDER BY churn_rate DESC;
        """)

    # --- pick an example platform & location one-hot if present ---
    platform_flag = first_col_with_prefix(conn, "platform_")  # e.g., platform_ios
    if platform_flag and col_exists(conn, "churned"):
        q(conn, f"churn_by_{platform_flag}", f"""
            SELECT "{platform_flag}" AS flag, COUNT(*) AS n, AVG(churned) AS churn_rate
            FROM feature_store
            GROUP BY "{platform_flag}";
        """)

    location_flag = first_col_with_prefix(conn, "location_")  # e.g., location_New_York
    if location_flag and col_exists(conn, "churned"):
        q(conn, f"churn_by_{location_flag}", f"""
            SELECT "{location_flag}" AS flag, COUNT(*) AS n, AVG(churned) AS churn_rate
            FROM feature_store
            GROUP BY "{location_flag}";
        """)

    # --- engineered features if present ---
    if col_exists(conn, "songs_per_hour"):
        q(conn, "top_songs_per_hour", """
            SELECT customer_id, songs_per_hour
            FROM feature_store
            WHERE songs_per_hour IS NOT NULL
            ORDER BY songs_per_hour DESC
            LIMIT 20;
        """)

    if col_exists(conn, "engagement_score"):
        q(conn, "top_engagement", """
            SELECT customer_id, engagement_score
            FROM feature_store
            ORDER BY engagement_score DESC
            LIMIT 20;
        """)

    # --- numeric distributions (choose a couple of typical features if they exist) ---
    for feature in ["weekly_hours", "average_session_length"]:
        if col_exists(conn, feature):
            q(conn, f"distribution_{feature}", f"""
                SELECT ROUND("{feature}", 1) AS {feature}_bin, COUNT(*) AS n
                FROM feature_store
                GROUP BY {feature}_bin
                ORDER BY {feature}_bin;
            """)

    # --- fetch a single customer’s feature row (use first available id) ---
    id_col = "customer_id" if col_exists(conn, "customer_id") else None
    if id_col:
        df_id = pd.read_sql_query(f"SELECT {id_col} AS id FROM feature_store LIMIT 1;", conn)
        if not df_id.empty:
            sample_id = df_id.iloc[0, 0]
            q(conn, "single_customer_row", f"""
                SELECT * FROM feature_store WHERE {id_col} = :cid;
            """, {"cid": sample_id})

    conn.close()
    print(f"\n✔ Queries complete. CSV outputs are in: {OUT_DIR}")

if __name__ == "__main__":
    main()
