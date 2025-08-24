# 7_FEATURE_STORE/getTrainingSet.py
from pathlib import Path
import sys

# --- make sure this folder is on sys.path ---
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from featureApi import get_training_dataframe  # now this works

OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    df = get_training_dataframe("churn_v1", 1)  # identical to churn_all v1 in your setup
    out = OUT / "training_set_churn_v1.csv"
    df.to_csv(out, index=False)
    print(f"[OK] wrote {out}  rows={len(df)}  cols={df.shape[1]}")

if __name__ == "__main__":
    main()
