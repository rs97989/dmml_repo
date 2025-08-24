# 7_FEATURE_STORE/feature_api.py
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB   = ROOT.parent / "6_DATA_TRANSFORMATION_AND_STORAGE" / "feature_store.db"

def connect():
    return sqlite3.connect(DB)

def list_features() -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query("SELECT * FROM fs_feature ORDER BY name", con)

def list_groups() -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query("SELECT * FROM fs_feature_group ORDER BY group_name, version", con)

def get_group_columns(group_name: str, version: int) -> list[str]:
    with connect() as con:
        df = pd.read_sql_query("""
          SELECT feature_name
          FROM fs_feature_group_feature
          WHERE group_name=? AND version=?
          ORDER BY ordinal
        """, con, params=[group_name, version])
    return df["feature_name"].tolist()

def get_group_keys(group_name: str, version: int) -> tuple[str|None, str|None]:
    with connect() as con:
        row = pd.read_sql_query("""
          SELECT primary_key, label
          FROM fs_feature_group
          WHERE group_name=? AND version=?
        """, con, params=[group_name, version])
    if row.empty:
        return None, None
    r = row.iloc[0]
    return r["primary_key"], r["label"]

def get_training_dataframe(group_name="churn_v1", version=1, where: str|None=None) -> pd.DataFrame:
    cols = get_group_columns(group_name, version)
    if not cols:
        raise ValueError(f"No columns found for group={group_name} v{version}")
    col_list = ", ".join([f'"{c}"' for c in cols])
    sql = f"SELECT {col_list} FROM feature_store"
    if where:
        sql += f" WHERE {where}"
    with connect() as con:
        return pd.read_sql_query(sql, con)

def get_online_vector(entity_id, group_name="churn_v1", version=1) -> dict:
    cols = get_group_columns(group_name, version)
    pk, _ = get_group_keys(group_name, version)
    if not pk:
        raise ValueError(f"Group {group_name} v{version} has no primary key.")
    col_list = ", ".join([f'"{c}"' for c in cols])
    sql = f'SELECT {col_list} FROM feature_store WHERE "{pk}" = ? LIMIT 1'
    with connect() as con:
        df = pd.read_sql_query(sql, con, params=[entity_id])
    return {} if df.empty else df.iloc[0].to_dict()

if __name__ == "__main__":
    # quick smoke test
    print(list_groups().head(3))
    feats = list_features().head(5)
    print("features sample:", feats.to_dict(orient="records"))
    X = get_training_dataframe().head(2)
    print("training sample shape:", X.shape)
    # try to fetch first id if present
    if "customer_id" in X.columns and not X.empty:
        rid = X.iloc[0]["customer_id"]
        print("online vector keys:", list(get_online_vector(rid))[:10], "…")
