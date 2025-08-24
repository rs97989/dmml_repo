from pathlib import Path
import sqlite3
import pandas as pd
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parent
DB   = ROOT.parent / "6_DATA_TRANSFORMATION_AND_STORAGE" / "feature_store.db"
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

def load_df(con, sql, params=None):
    return pd.read_sql_query(sql, con, params=params or {})

def feature_stats(con) -> pd.DataFrame:
    # Build quick stats per column from the live feature table
    cols = load_df(con, "PRAGMA table_info(feature_store)")["name"].tolist()
    rows = []
    for c in cols:
        # missing rate + distinct
        miss = load_df(con, f'''
            SELECT AVG(CASE WHEN "{c}" IS NULL THEN 1.0 ELSE 0.0 END) AS missing_rate,
                   COUNT(DISTINCT "{c}") AS distinct_count
            FROM feature_store
        ''').iloc[0]
        # numeric moments (mean/std); NULL if not numeric
        num = load_df(con, f'''
            SELECT AVG(CASE WHEN typeof("{c}") IN ('integer','real') THEN "{c}" END) AS mean,
                   -- sample stddev approximation using variance formula
                   AVG(CASE WHEN typeof("{c}") IN ('integer','real') THEN ("{c}" - 
                         (SELECT AVG(CASE WHEN typeof("{c}") IN ('integer','real') THEN "{c}" END) 
                          FROM feature_store)
                       ) * ("{c}" - 
                         (SELECT AVG(CASE WHEN typeof("{c}") IN ('integer','real') THEN "{c}" END) 
                          FROM feature_store)
                       ) END) AS var
            FROM feature_store
        ''').iloc[0]
        std = (float(num["var"]) ** 0.5) if pd.notna(num["var"]) else None
        rows.append({
            "name": c,
            "missing_rate": float(miss["missing_rate"] or 0.0),
            "distinct_count": int(miss["distinct_count"] or 0),
            "mean": None if pd.isna(num["mean"]) else float(num["mean"]),
            "std": None if std is None else float(std),
        })
    return pd.DataFrame(rows)

def export():
    if not DB.exists():
        raise SystemExit(f"DB not found: {DB}")

    con = sqlite3.connect(DB)

    # registry tables
    feats   = load_df(con, "SELECT * FROM fs_feature ORDER BY name")
    groups  = load_df(con, "SELECT * FROM fs_feature_group ORDER BY group_name, version")
    members = load_df(con, """
        SELECT group_name, version, feature_name, ordinal
        FROM fs_feature_group_feature
        ORDER BY group_name, version, ordinal
    """)

    # compute live stats and merge with metadata
    stats = feature_stats(con)
    feats_doc = feats.merge(stats, on="name", how="left")

    # write CSVs
    feats_doc.to_csv(DOCS / "features.csv", index=False)
    groups.to_csv(DOCS / "feature_groups.csv", index=False)
    members.to_csv(DOCS / "group_membership.csv", index=False)

    # also render a concise Markdown summary
    md = []
    md.append(f"# Feature Store Documentation\n")
    md.append(f"_generated: {datetime.utcnow().isoformat()}Z_\n")
    md.append(f"**DB:** `{DB}`\n")

    md.append("## Feature Groups\n")
    for _, g in groups.iterrows():
        cnt = (members["group_name"].eq(g["group_name"]) & members["version"].eq(g["version"])).sum()
        md.append(f"- **{g['group_name']} v{g['version']}** — pk=`{g['primary_key']}`"
                  + (f", label=`{g['label']}`" if pd.notna(g['label']) else "")
                  + f", features={cnt}\n")

    md.append("\n## Example: churn_v1 v1 membership (first 25)\n")
    gm = members[(members.group_name=="churn_v1") & (members.version==1)].head(25)
    if not gm.empty:
        for _, r in gm.iterrows():
            md.append(f"- {r['feature_name']}\n")
    else:
        md.append("_Group churn_v1 v1 not found._\n")

    md.append("\n## Files\n")
    md.append(f"- `features.csv` — feature metadata + live stats\n")
    md.append(f"- `feature_groups.csv` — groups and versions\n")
    md.append(f"- `group_membership.csv` — ordered features per group/version\n")

    (DOCS / "README.md").write_text("\n".join(md), encoding="utf-8")
    con.close()

    print(f"[OK] Wrote docs to: {DOCS}")
    print(" - features.csv")
    print(" - feature_groups.csv")
    print(" - group_membership.csv")
    print(" - README.md")

if __name__ == "__main__":
    export()
