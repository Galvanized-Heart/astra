from pathlib import Path
import pandas as pd

# ---------------- Project paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

# Source embeddings root (must already exist).
# Try env override first; otherwise set the correct mount below.
SRC_EMB_PATH = Path(
    # os.environ.get("ASTRA_EMB_PATH", "/gpfs/fs0/scratch/m/mahadeva/maxkirby/astra-embeddings")
    "/gpfs/fs0/scratch/m/mahadeva/maxkirby/astra-embeddings"  # <- change to gs0 only if it actually exists on this node
)
if not SRC_EMB_PATH.exists():
    raise FileNotFoundError(
        f"SRC_EMB_PATH does not exist on this node: {SRC_EMB_PATH}\n"
        "Tip: run `ls -ld /gpfs/fs0 /gpfs/gs0` and point to the mounted one."
    )

# All outputs go to a subdir you own (safe to mkdir)
DST_DIR = SRC_EMB_PATH / "cpipred_splits"
DST_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Config ----------------
INPUT_GLOB_PATTERNS = ["fold_*_train.csv", "fold_*_valid.csv"]
SOURCE_SEQ_COL = "protein_sequence"
SOURCE_SMILES_COL = "ligand_smiles"
PARAM_COLS = ["kcat", "Ki", "KM"]

def read_any_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")

def make_param_frame(df: pd.DataFrame, param: str) -> pd.DataFrame:
    if param not in df.columns:
        return pd.DataFrame(columns=["CMPD_SMILES", "SEQ", param])

    col = df[param]
    mask_nonempty = col.notna() & (col.astype(str).str.strip() != "")
    numeric = pd.to_numeric(col, errors="coerce")
    mask_numeric = numeric.notna()
    mask = mask_nonempty & mask_numeric

    out = df.loc[mask, [SOURCE_SMILES_COL, SOURCE_SEQ_COL, param]].copy()
    out.rename(columns={SOURCE_SMILES_COL: "CMPD_SMILES", SOURCE_SEQ_COL: "SEQ"}, inplace=True)
    out[param] = pd.to_numeric(out[param], errors="coerce")
    out.drop_duplicates(inplace=True)
    return out

def process_file(csv_path: Path) -> None:
    df = read_any_csv(csv_path)
    for param in PARAM_COLS:
        out_df = make_param_frame(df, param)
        out_name = f"Astra_{csv_path.stem}_{param}.csv"
        out_path = DST_DIR / out_name
        out_df.to_csv(out_path, index=True)  # set index=False if you don't want the index column
        print(f"→ {out_path.relative_to(SRC_EMB_PATH)}: {len(out_df)} rows")

def main():
    any_found = False
    for pattern in INPUT_GLOB_PATTERNS:
        for csv_path in sorted(SRC_EMB_PATH.rglob(pattern)):  # recursive
            any_found = True
            print(f"Processing {csv_path.relative_to(SRC_EMB_PATH)} ...")
            process_file(csv_path)
    if not any_found:
        print(f"No input files found under {SRC_EMB_PATH} matching {INPUT_GLOB_PATTERNS}")

if __name__ == "__main__":
    main()
