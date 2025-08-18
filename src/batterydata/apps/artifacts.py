"""
Artifacts I/O utilities for batterydata.

Responsibilities
---------------
- Discover preprocessed cell artifacts in `artifacts/<cell>/`.
- Load manifests and tables (parquet with CSV fallback).
- Save KPI tables next to each cell's artifacts with provenance metadata.

This module is UI-agnostic and can be reused by notebooks, CLIs, or GUIs.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import pandas as pd
import json
import datetime as dt


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def global_kpi_dir(self) -> Path:
        return self.artifacts_dir / "kpi"


class ArtifactStore:
    """
    Provides read/write access to preprocessed artifacts and KPI outputs.
    """

    def __init__(self, project_root: Path):
        self.paths = ProjectPaths(project_root=project_root)
        self.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.paths.global_kpi_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Discovery ----------

    def discover_cells(self) -> List[Path]:
        """Return cell directories that contain a manifest.yaml."""
        art = self.paths.artifacts_dir
        if not art.exists():
            return []
        return sorted([p for p in art.iterdir() if (p / "manifest.yaml").exists()])

    # ---------- Manifest / blocks ----------

    def load_manifest(self, cell_dir: Path) -> Dict:
        """Load manifest.yaml inside a cell directory."""
        with open(cell_dir / "manifest.yaml", "r") as f:
            return yaml.safe_load(f)

    def load_blocks(self, cell_dir: Path) -> List[Dict]:
        """Load blocks.json inside a cell directory; empty list if missing."""
        fp = cell_dir / "blocks.json"
        if not fp.exists():
            return []
        with open(fp, "r") as f:
            return json.load(f)

    # ---------- Table I/O with fallback ----------

    def _read_parquet_or_csv(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".parquet":
            try:
                return pd.read_parquet(path)
            except Exception:
                # try a sibling CSV with same stem
                return pd.read_csv(path.with_suffix(".csv"))
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        # unknown extension: try parquet then csv
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path.with_suffix(".csv"))

    def load_table_from_manifest(self, manifest: Dict, key: str) -> pd.DataFrame:
        """Load a table using a key from manifest['artifacts'] (e.g., 'segmented_data', 'doe_table')."""
        rel = manifest["artifacts"][key]
        return self._read_parquet_or_csv(self.paths.project_root / rel)

    # ---------- KPI save ----------

    def save_kpi_table(
        self,
        cell_dir: Path,
        df: pd.DataFrame,
        meta: Dict,
        basename: str = "KPI_1",
    ) -> Tuple[Path, Path]:
        """
        Save a KPI table next to the cell artifacts:

        - artifacts/<cell>/kpi/<basename>.parquet (fallback CSV)
        - artifacts/<cell>/kpi/<basename>.metadata.yaml

        Returns
        -------
        (table_path, metadata_path)
        """
        kpi_dir = cell_dir / "kpi"
        kpi_dir.mkdir(exist_ok=True)

        parquet_path = kpi_dir / f"{basename}.parquet"
        try:
            df.to_parquet(parquet_path, index=False)
            table_path = parquet_path
            meta["format"] = "parquet"
        except Exception:
            csv_path = kpi_dir / f"{basename}.csv"
            df.to_csv(csv_path, index=False)
            table_path = csv_path
            meta["format"] = "csv"

        meta.setdefault("created_at", dt.datetime.now().isoformat(timespec="seconds"))
        meta_path = kpi_dir / f"{basename}.metadata.yaml"
        with open(meta_path, "w") as f:
            yaml.safe_dump(meta, f, sort_keys=False)

        return table_path, meta_path

    # ---------- Global KPI save (optional) ----------

    def save_global_kpi_table(self, df: pd.DataFrame, basename: str = "KPI_1_all_cells") -> Path:
        """
        Save a combined KPI table across cells into artifacts/kpi/.
        Uses parquet with CSV fallback.
        """
        out_parq = self.paths.global_kpi_dir / f"{basename}.parquet"
        try:
            df.to_parquet(out_parq, index=False)
            return out_parq
        except Exception:
            out_csv = self.paths.global_kpi_dir / f"{basename}.csv"
            df.to_csv(out_csv, index=False)
            return out_csv
