# 5_DATA_PREPARATION/prepareData.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json
import warnings
import re

import numpy as np
import pandas as pd

# non-interactive plotting (no GUI needed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Paths & config ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT     = PROJECT_ROOT / "3_RAW_DATA_STORAGE" / "raw_data"
OUT_DIR      = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = ["kaggle", "source"]

ID_COL          = "customer_id"
TARGET_COL      = "churned"
DATE_CANDIDATES = ["signup_date", "created_at", "registration_date"]

FORCE_INT_COLS   = [
    "age", TARGET_COL,
    "weekly_songs_played", "weekly_unique_songs",
    "num_subscription_pauses", "num_favorite_artists",
    "num_platform_friends", "num_playlists_created",
    "num_shared_playlists", "notifications_clicked",
]
FORCE_FLOAT_COLS = ["weekly_hours", "average_session_length", "song_skip_rate"]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------- Utils ----------------
def ts_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def latest_csv_for_source(source: str) -> Path | None:
    base = RAW_ROOT / source / "csv"
    if not base.exists():
        return None
    ts_dirs = sorted(
        [p for p in base.glob("ingestion_ts=*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime
    )
    if not ts_dirs:
        return None
    csvs = sorted(ts_dirs[-1].glob("*.csv"))
    return csvs[0] if csvs else None

def _safe_to_datetime(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if dt.notna().mean() < 0.25 and pd.api.types.is_integer_dtype(series):
        dt2 = pd.to_datetime(series, errors="coerce", unit="s", utc=True)
        if dt2.notna().mean() > dt.notna().mean():
            dt = dt2
    return dt

def load_sources(sources: list[str]) -> list[pd.DataFrame]:
    frames = []
    for src in sources:
        p = latest_csv_for_source(src)
        if p is None:
            print(f"[WARN] No latest file for source={src}")
            continue
        df = pd.read_csv(p)
        df["__source__"] = src
        frames.append(df)
        print(f"[INFO] Loaded {src}: {p.name} rows={len(df)}")
    return frames

def align_schema(frames: list[pd.DataFrame]) -> pd.DataFrame:
    all_cols = sorted(set().union(*[f.columns for f in frames]))
    frames2 = [f.reindex(columns=all_cols) for f in frames]
    return pd.concat(frames2, ignore_index=True)

# ---------------- Name sanitizers ----------------
_name_re = re.compile(r"[^A-Za-z0-9_]+")

def sanitize_token(text: str) -> str:
    """
    Keep letters/digits/underscore; replace others with '_', collapse repeats,
    strip leading/trailing '_', and prefix if starts with a digit.
    """
    s = _name_re.sub("_", str(text).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "NA"
    if s[0].isdigit():
        s = f"C_{s}"
    return s

def sanitize_and_dedup_columns(cols: list[str]) -> list[str]:
    seen = {}
    out = []
    for c in cols:
        base = sanitize_token(c)
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out

# ---------------- Cleaning ----------------
def enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype(str)

    for c in FORCE_INT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")

    for c in FORCE_FLOAT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    for c in DATE_CANDIDATES:
        if c in df.columns:
            df[c] = _safe_to_datetime(df[c])
    return df

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c == TARGET_COL:  # do not invent target label
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            if s.isna().any():
                df[c] = s.fillna(s.median())
        else:
            if s.isna().any():
                mode = s.mode(dropna=True)
                df[c] = s.fillna(mode.iloc[0] if not mode.empty else "unknown")
    return df

def winsorize_iqr(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in columns:
        s = pd.to_numeric(df[c], errors="coerce")
        q1 = s.quantile(0.25); q3 = s.quantile(0.75); iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lo = q1 - 1.5 * iqr; hi = q3 + 1.5 * iqr
        df[c] = s.clip(lower=lo, upper=hi)
    return df

def zscore_scale(df: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    for c in numeric_cols:
        s = df[c].astype(float)
        std = s.std()
        if std and not np.isnan(std) and std != 0:
            df[c] = (s - s.mean()) / std
    return df

def one_hot_encode_safe(df: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
    """
    One-hot encode object/categorical columns with *sanitized* dummy column names.
    We do NOT change the original categorical values in df; only the dummy names.
    """
    cat_cols = [c for c in df.columns
                if (df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c]))
                and c not in exclude]
    if not cat_cols:
        return df

    out = df.drop(columns=cat_cols)

    for c in cat_cols:
        # Build dummies from stringified values (without mutating original column)
        vals = df[c].astype(str).fillna("unknown")
        # Create raw dummies first (so categories are preserved)
        raw = pd.get_dummies(vals, prefix=c, prefix_sep="_", dummy_na=False)
        # Sanitize resulting column names (fix spaces, punctuation, collisions)
        raw.columns = sanitize_and_dedup_columns(list(raw.columns))
        # Attach
        out = pd.concat([out, raw], axis=1)

    return out

# ---------------- EDA ----------------
def eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    try:
        desc = df.describe(include="all", datetime_is_numeric=True).transpose()
    except TypeError:
        desc = df.describe(include="all").transpose()
    desc.to_csv(OUT_DIR / "eda_summary.csv")
    return desc

def plot_histograms(df: pd.DataFrame, limit: int = 12) -> list[Path]:
    paths = []
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric_cols[:limit]:
        plt.figure()
        df[c].dropna().hist(bins=30)
        plt.title(f"Histogram: {c}")
        plt.xlabel(c); plt.ylabel("count")
        plt.tight_layout()
        fp = OUT_DIR / f"hist_{c}.png"
        plt.savefig(fp); plt.close()
        paths.append(fp)
    return paths

def plot_boxplots(df: pd.DataFrame, limit: int = 12) -> list[Path]:
    paths = []
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric_cols[:limit]:
        plt.figure()
        df[[c]].plot(kind="box", legend=False)
        plt.title(f"Boxplot: {c}")
        plt.tight_layout()
        fp = OUT_DIR / f"box_{c}.png"
        plt.savefig(fp); plt.close()
        paths.append(fp)
    return paths

def write_html_gallery(images: list[Path]) -> Path:
    html = ["<html><head><meta charset='utf-8'><title>EDA Gallery</title></head><body>",
            "<h1>EDA Plots</h1>"]
    for img in images:
        html.append(f"<div style='margin:12px 0'><h3>{img.name}</h3>"
                    f"<img src='{img.name}' style='max-width:100%'></div>")
    html.append("</body></html>")
    fp = OUT_DIR / "eda_gallery.html"
    fp.write_text("\n".join(html), encoding="utf-8")
    return fp

# ---------------- Pipeline ----------------
def main():
    warnings.filterwarnings("ignore")

    # 1) Load latest from kaggle + source
    frames = load_sources(SOURCES)
    if not frames:
        raise SystemExit("No raw files found in 3_RAW_DATA_STORAGE/raw_data/*/csv")
    df_raw = align_schema(frames)
    print(f"[INFO] Combined shape: {df_raw.shape}")

    # 2) Enforce types
    df = enforce_types(df_raw.copy())

    # 3) Dedupe (full row + by ID if present)
    before = len(df)
    df = df.drop_duplicates()
    if ID_COL in df.columns:
        df = df.drop_duplicates(subset=[ID_COL], keep="first")
    print(f"[INFO] Dedupe: {before} -> {len(df)} rows")

    # 4) Impute missing, then winsorize outliers
    df = impute_missing(df)
    df = winsorize_iqr(df)

    # 5) Save clean (human-readable) CSV
    clean_csv = OUT_DIR / "cleaned_dataset.csv"
    df.to_csv(clean_csv, index=False)
    print(f"[OK] Clean (pre-encoding) -> {clean_csv}")

    # 6) Build model-ready matrix
    exclude = [ID_COL, TARGET_COL, "__source__"] + [c for c in DATE_CANDIDATES if c in df.columns]
    df_model = one_hot_encode_safe(df, exclude=exclude)
    df_model = zscore_scale(df_model, exclude=exclude)

    # Final safety: sanitize & dedup ALL column names (no spaces, no collisions)
    df_model.columns = sanitize_and_dedup_columns(list(df_model.columns))

    matrix_parquet = OUT_DIR / "model_matrix.parquet"
    matrix_csv     = OUT_DIR / "model_matrix.csv"
    df_model.to_parquet(matrix_parquet, index=False)  # requires pyarrow or fastparquet
    df_model.to_csv(matrix_csv, index=False)
    print(f"[OK] Model matrix -> {matrix_parquet} / {matrix_csv}")

    # 7) EDA: PNGs + HTML gallery
    eda_summary(df)
    imgs = []
    imgs += plot_histograms(df)
    imgs += plot_boxplots(df)
    gallery = write_html_gallery(imgs)
    print(f"[OK] EDA images + gallery -> {gallery}")

    # 8) Manifest
    manifest = {
        "prepared_at_utc": ts_utc(),
        "sources": SOURCES,
        "raw_rows_total": int(len(df_raw)),
        "clean_rows": int(len(df)),
        "clean_file": str(clean_csv),
        "model_matrix_parquet": str(matrix_parquet),
        "model_matrix_csv": str(matrix_csv),
        "eda_gallery": str(gallery),
    }
    with open(OUT_DIR / "prepare_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("[DONE] Data preparation complete.")

if __name__ == "__main__":
    main()
