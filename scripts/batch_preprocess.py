#!/usr/bin/env python3
"""
Batch Preprocess Script for LCO Cells (simple logging)

- Reads cells from LCO_cells.yaml
- Runs preprocessing (segmentation + DoE)
- Saves artifacts to project-root/artifacts/<cell_name>/
- Prints concise progress; records a minimal CSV log at the end.
"""

import sys, os, json, yaml, datetime as dt, time, traceback
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# --- Make src/ importable (script is sibling of src/) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from batterydata.parsers.basytec_reader import BasyTecReader
from batterydata.pipeline.preprocessor import Preprocessor

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG = ARTIFACTS_DIR / "preprocess_log.csv"


def save_tables(cell_dir: Path, tables: dict, blocks: list, root: Path) -> None:
    """Save segmented/doe as parquet (fallback to CSV), metadata as YAML, blocks as JSON."""
    cell_dir.mkdir(parents=True, exist_ok=True)

    seg_path_parq = cell_dir / "raw.parquet"
    doe_path_parq = cell_dir / "doe.parquet"
    seg_path_csv = cell_dir / "raw.csv"
    doe_path_csv = cell_dir / "doe.csv"

    # 1) segmented_data + DoE → Parquet (fallback CSV if engine missing)
    try:
        tables["segmented_data"].to_parquet(seg_path_parq, index=False)
        tables["doe_table"].to_parquet(doe_path_parq, index=False)
    except Exception:
        # Fallback to CSV for maximum portability
        tables["segmented_data"].to_csv(seg_path_csv, index=False)
        tables["doe_table"].to_csv(doe_path_csv, index=False)

    # 2) metadata → YAML (handles lists/dicts safely)
    meta_yaml = cell_dir / "metadata.yaml"
    meta_obj = tables.get("metadata")
    if isinstance(meta_obj, pd.DataFrame):
        meta_content = meta_obj.to_dict(orient="records")
    else:
        # if your preprocessor returns a dict already, great
        try:
            meta_content = meta_obj
        except Exception:
            meta_content = {}

    with open(meta_yaml, "w") as f:
        yaml.safe_dump(meta_content, f, sort_keys=False)

    # 3) blocks → JSON
    with open(cell_dir / "blocks.json", "w") as f:
        json.dump(blocks, f, indent=2)

    # 4) manifest → YAML (relative paths)
    manifest = {"cell_name": cell_dir.name, "artifacts": {}}
    # Prefer parquet entries if they exist, otherwise csv
    manifest["artifacts"]["segmented_data"] = (
        str(seg_path_parq.relative_to(root)) if seg_path_parq.exists()
        else str(seg_path_csv.relative_to(root))
    )
    manifest["artifacts"]["doe_table"] = (
        str(doe_path_parq.relative_to(root)) if doe_path_parq.exists()
        else str(doe_path_csv.relative_to(root))
    )
    manifest["artifacts"]["metadata"] = str(meta_yaml.relative_to(root))
    manifest["artifacts"]["blocks"] = str((cell_dir / "blocks.json").relative_to(root))

    with open(cell_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def main():
    # --- Load config ---
    cfg_path = PROJECT_ROOT / "LCO_cells.yaml"
    if not cfg_path.exists():
        print(f"[ERR] Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    base_raw_path = Path(config["path_to_raw_data"])
    cells = [[c["name"], base_raw_path / c["path"]] for c in config["LCO_cells"]]

    print(f"Found {len(cells)} cells. Artifacts → {ARTIFACTS_DIR}")

    log_rows = []  # collect rows, build DataFrame at end
    t0 = time.time()

    for name, file_path in tqdm(cells, desc="Preprocessing", unit="cell"):
        ts = dt.datetime.now().isoformat(timespec="seconds")
        t_cell = time.time()

        print(f"→ {name}: loading {file_path}")
        try:
            # Load
            reader = BasyTecReader(filepath=str(file_path), sample_id=name)
            result = reader.read()
            df = result["data"]
            raw_rows = len(df)

            # Preprocess
            preproc = Preprocessor(df, cell_capacity_ah=1.3)
            seg, doe, blocks = preproc.process_with_blocks(
                min_block_len=4, max_block_len=4, min_repeats=2
            )
            tables = preproc.export_tables()

            # Save
            cell_dir = ARTIFACTS_DIR / name
            save_tables(cell_dir, tables, blocks, PROJECT_ROOT)

            dur = round(time.time() - t_cell, 2)
            print(f"✓ {name}: OK in {dur}s (raw={raw_rows}, seg={len(tables['segmented_data'])}, doe={len(tables['doe_table'])})")
            log_rows.append({
                "ts": ts, "cell": name, "status": "OK",
                "raw_rows": raw_rows,
                "seg_rows": len(tables["segmented_data"]),
                "doe_rows": len(tables["doe_table"]),
                "duration_s": dur,
                "msg": ""
            })

        except Exception as e:
            dur = round(time.time() - t_cell, 2)
            print(f"✗ {name}: ERROR in {dur}s → {type(e).__name__}: {e}")
            # optional: print traceback for quick debugging
            traceback.print_exc(limit=1)
            log_rows.append({
                "ts": ts, "cell": name, "status": "ERROR",
                "raw_rows": None, "seg_rows": None, "doe_rows": None,
                "duration_s": dur,
                "msg": f"{type(e).__name__}: {e}"
            })

    # Save run log once (avoids column mismatch)
    log_df = pd.DataFrame(log_rows, columns=[
        "ts", "cell", "status", "raw_rows", "seg_rows", "doe_rows", "duration_s", "msg"
    ])
    # If an old log exists, append to it
    if RUN_LOG.exists():
        old = pd.read_csv(RUN_LOG)
        log_df = pd.concat([old, log_df], ignore_index=True)

    log_df.to_csv(RUN_LOG, index=False)

    print(f"\nDone in {round(time.time()-t0,2)}s. Log saved → {RUN_LOG}")


if __name__ == "__main__":
    main()
