# 4_DATA_VALIDATION/validateData.py
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

RAW_ROOT = Path(__file__).resolve().parents[1] / "3_RAW_DATA_STORAGE" / "raw_data"
OUT_DIR  = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Optional project-specific rules -----------------
# Add column-specific checks here if you want stricter validation.
RULES: Dict[str, Dict[str, Any]] = {
    # "age": {"min": 0, "max": 120},
    # "churned": {"allowed_values": [0, 1]},
    # "signup_date": {"format": "datetime"},
}

# ----------------- Helpers -----------------
def ts_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def latest_csv_for_source(source: str) -> Optional[Path]:
    """
    Return the newest CSV for a given source (kaggle/source) under csv/.
    """
    base = RAW_ROOT / source / "csv"
    if not base.exists():
        return None
    ts_dirs = sorted(
        [p for p in base.glob("ingestion_ts=*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime
    )
    if not ts_dirs:
        return None
    latest_dir = ts_dirs[-1]
    csvs = sorted(latest_dir.glob("*.csv"))
    return csvs[0] if csvs else None

def _safe_to_datetime(s: pd.Series) -> pd.Series:
    # pandas now infers datetime formats by default; no need for infer_datetime_format
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if dt.notna().mean() < 0.25 and pd.api.types.is_integer_dtype(s):
        # Fallback: treat integers as epoch seconds if plain parse failed
        dt2 = pd.to_datetime(s, errors="coerce", unit="s", utc=True)
        if dt2.notna().mean() > dt.notna().mean():
            dt = dt2
    return dt

def _iqr_outlier_mask(x: pd.Series) -> pd.Series:
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=x.index)
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return (x < lo) | (x > hi)

def _top_freq(s: pd.Series, k: int = 10) -> str:
    vc = s.value_counts(dropna=True).head(k)
    return json.dumps(vc.to_dict(), ensure_ascii=False)

def _guess_id_columns(cols: List[str]) -> List[str]:
    return [c for c in cols if "id" in c.lower()]

# ----------------- Validation core -----------------
def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        is_num = pd.api.types.is_numeric_dtype(s)
        is_int = pd.api.types.is_integer_dtype(s)
        is_float = pd.api.types.is_float_dtype(s)
        n = len(s)

        missing = int(s.isna().sum())
        missing_pct = round(100 * missing / max(1, n), 2)
        nunique = int(s.nunique(dropna=True))

        rec: Dict[str, Any] = {
            "column": col,
            "dtype": dtype,
            "is_numeric": bool(is_num),
            "is_integer": bool(is_int),
            "is_float": bool(is_float),
            "n_rows": n,
            "missing": missing,
            "missing_pct": missing_pct,
            "n_unique": nunique,
        }

        if is_num:
            s_num = pd.to_numeric(s, errors="coerce")
            rec.update({
                "min": float(np.nanmin(s_num)) if s_num.notna().any() else None,
                "p25": float(s_num.quantile(0.25)) if s_num.notna().any() else None,
                "mean": float(s_num.mean()) if s_num.notna().any() else None,
                "p75": float(s_num.quantile(0.75)) if s_num.notna().any() else None,
                "max": float(np.nanmax(s_num)) if s_num.notna().any() else None,
                "std": float(s_num.std()) if s_num.notna().any() else None,
                "negatives": int((s_num < 0).sum(skipna=True)),
                "zeros": int((s_num == 0).sum(skipna=True)),
            })
            out_mask = _iqr_outlier_mask(s_num.dropna())
            rec["iqr_outliers"] = int(out_mask.sum())
        else:
            rec["top_10_freq"] = _top_freq(s.dropna())

        # Date-format check if rule demands it OR column name hints at dates
        wants_date_check = RULES.get(col, {}).get("format") == "datetime" or "date" in col.lower()
        if wants_date_check:
            dt = _safe_to_datetime(s)
            rec["date_parseable_pct"] = round(100 * dt.notna().mean(), 2)
            if dt.notna().any():
                rec["date_min"] = dt.min().isoformat()
                rec["date_max"] = dt.max().isoformat()
            else:
                rec["date_min"] = rec["date_max"] = None

        # Rules: allowed values / min-max ranges
        r = RULES.get(col)
        if r:
            if "allowed_values" in r:
                allowed = set(r["allowed_values"])
                invalid_mask = ~s.isin(list(allowed))
                rec["rule_invalid_allowed_values"] = int(invalid_mask.sum())
            if "min" in r or "max" in r:
                s_num = pd.to_numeric(s, errors="coerce") if not is_num else s
                bad = pd.Series(False, index=s.index)
                if "min" in r:
                    bad = bad | (s_num < r["min"])
                if "max" in r:
                    bad = bad | (s_num > r["max"])
                rec["rule_out_of_range"] = int(bad.sum(skipna=True))

        rows.append(rec)

    return pd.DataFrame(rows)

def duplicate_reports(df: pd.DataFrame) -> Dict[str, Any]:
    # Full-row duplicates
    full_dups_mask = df.duplicated(keep=False)
    full_dups_count = int(full_dups_mask.sum())

    # ID-based duplicates (heuristic)
    id_cols = _guess_id_columns(df.columns.tolist())
    id_dups_summary = []
    for c in id_cols:
        dcnt = int(df[c].duplicated(keep=False).sum())
        if dcnt > 0:
            id_dups_summary.append({"id_column": c, "duplicate_rows": dcnt})

    return {
        "full_dups_count": full_dups_count,
        "full_dups_rows": df[full_dups_mask].head(1000),  # cap sample
        "id_dups_summary": pd.DataFrame(id_dups_summary) if id_dups_summary else pd.DataFrame(columns=["id_column","duplicate_rows"]),
    }

def build_summary_issues(profile: pd.DataFrame, dup_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Compact table listing key issues per column so reviewers can see hotspots fast.
    """
    issues = []
    for _, r in profile.iterrows():
        col = r["column"]
        # Missing
        if r["missing"] > 0:
            issues.append({"column": col, "issue": "missing_values", "count": int(r["missing"])})
        # Negatives, zeros (numeric only)
        if r.get("negatives", 0):
            issues.append({"column": col, "issue": "negative_values", "count": int(r["negatives"])})
        if r.get("zeros", 0):
            issues.append({"column": col, "issue": "zeros", "count": int(r["zeros"])})
        # Outliers
        if r.get("iqr_outliers", 0):
            issues.append({"column": col, "issue": "iqr_outliers", "count": int(r["iqr_outliers"])})
        # Rules
        if r.get("rule_out_of_range", 0):
            issues.append({"column": col, "issue": "rule_out_of_range", "count": int(r["rule_out_of_range"])})
        if r.get("rule_invalid_allowed_values", 0):
            issues.append({"column": col, "issue": "invalid_allowed_values", "count": int(r["rule_invalid_allowed_values"])})
        # Date parse
        if "date_parseable_pct" in r and r["date_parseable_pct"] < 100:
            bad = int(round(r["n_rows"] * (100 - r["date_parseable_pct"]) / 100))
            issues.append({"column": col, "issue": "unparseable_dates", "count": bad})

    # Duplicates (dataset-level)
    if dup_info["full_dups_count"] > 0:
        issues.append({"column": "(dataset)", "issue": "full_row_duplicates", "count": int(dup_info["full_dups_count"])})

    # ID duplicates
    if not dup_info["id_dups_summary"].empty:
        for _, row in dup_info["id_dups_summary"].iterrows():
            issues.append({"column": row["id_column"], "issue": "id_duplicates", "count": int(row["duplicate_rows"])})

    return pd.DataFrame(issues, columns=["column", "issue", "count"]).sort_values(["count"], ascending=False)

def validate_file(source: str, file_path: Path) -> dict:
    df = pd.read_csv(file_path)

    profile = profile_dataframe(df)
    dup_info = duplicate_reports(df)
    summary_issues = build_summary_issues(profile, dup_info)

    # Compact CSV summary
    summary_csv = OUT_DIR / f"validation_{source}.csv"
    profile.to_csv(summary_csv, index=False)

    # Rich Excel report
    report_xlsx = OUT_DIR / f"validation_{source}.xlsx"
    with pd.ExcelWriter(report_xlsx, engine="xlsxwriter") as xw:
        # High-level summary
        high_level = pd.DataFrame(
            [
                {"metric": "rows", "value": len(df)},
                {"metric": "columns", "value": df.shape[1]},
                {"metric": "full_row_duplicates", "value": dup_info["full_dups_count"]},
                {"metric": "generated_at_utc", "value": ts_utc()},
                {"metric": "source_file", "value": str(file_path)},
            ]
        )
        high_level.to_excel(xw, index=False, sheet_name="summary")

        # Column-level profile
        profile.to_excel(xw, index=False, sheet_name="column_profile")

        # Numeric-only stats
        (profile[profile["is_numeric"] == True]).to_excel(xw, index=False, sheet_name="numeric_stats")

        # Categorical top frequencies
        cat_profile = profile[profile["is_numeric"] == False].copy()
        if not cat_profile.empty and "top_10_freq" in cat_profile.columns:
            cat_profile["top_10_freq"] = cat_profile["top_10_freq"].fillna("{}")
        cat_profile.to_excel(xw, index=False, sheet_name="categorical")

        # Duplicates (samples + ID summary)
        dup_info["full_dups_rows"].to_excel(xw, index=False, sheet_name="duplicate_rows")
        dup_info["id_dups_summary"].to_excel(xw, index=False, sheet_name="id_duplicates")

        # Issues rollup
        summary_issues.to_excel(xw, index=False, sheet_name="Summary_Issues")

    return {
        "source": source,
        "status": "success",
        "file": str(file_path),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicates": int(dup_info["full_dups_count"]),
        "report_csv": str(summary_csv),
        "report_xlsx": str(report_xlsx),
    }

# ----------------- CLI -----------------
def main():
    sources = ["kaggle", "source"]
    run_results = []

    for src in sources:
        csv_path = latest_csv_for_source(src)
        if csv_path is None:
            run_results.append({"source": src, "status": "missing"})
            continue
        try:
            res = validate_file(src, csv_path)
            run_results.append(res)
        except Exception as e:
            run_results.append({"source": src, "status": "failed", "error": str(e)})

    # Append a summary log line
    log_path = OUT_DIR / "validation_summary.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{ts_utc()} - {run_results}\n")

    print("Validation complete. See 4_DATA_VALIDATION/outputs for reports and logs.")

if __name__ == "__main__":
    main()
