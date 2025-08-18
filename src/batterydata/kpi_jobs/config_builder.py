from __future__ import annotations

import os
import re
import sys
import json
import uuid
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Iterable

try:
    from pydantic import BaseModel, Field, validator, root_validator
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Pydantic is required. Install with `pip install pydantic`."
    ) from e

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOB_SCHEMA_VERSION = 1
DEFAULT_JOIN_KEY = "step_number"
JOB_FILE_PREFIX = "kpi_"
DEFAULT_TABLE_FILENAME = "kpi_table.parquet"
DEFAULT_LOG_FILENAME = "run.log"
DEFAULT_SUMMARY_FILENAME = "summary.json"

# Optional: lightweight built-in KPI presets; you can extend/override at runtime.
KPI_PRESETS: Dict[str, List[str]] = {
    "cycle_core": [
        "QChargeAh", "QDischargeAh",
        "EChargeWh", "EDischargeWh",
        "CoulombicEfficiency", "EnergyEfficiency",
        "VChargeAvg", "VDischargeAvg",
        "ChargeTotalDurationSeconds", "ChargeCVDurationSeconds",
        "ChargeCVTimeRatio", "ChargeCVCapacityRatio",
        "CCVTotalMismatchPct", "QChargeAhCV", "QChargeAhCC",
        "QChargeAhByCCCV",
    ],
}

# ---------------------------------------------------------------------------
# KPI Registry Hook (optional but recommended)
# ---------------------------------------------------------------------------

class KPIRegistry:
    """
    Resolve KPI names to implementation classes/modules.
    Plug your real registry here (e.g., introspect batterydata.pipeline.kpi_engine).
    """

    def __init__(self, available_names: Optional[Iterable[str]] = None) -> None:
        # If not provided, accept any non-empty string (soft validation).
        self._names = set(available_names) if available_names else None

    def is_valid(self, name: str) -> bool:
        if not name or not isinstance(name, str):
            return False
        if self._names is None:
            return True
        return name in self._names

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class Selection(BaseModel):
    blocks: List[int] = Field(..., description="Block IDs to process.")
    cycles: Any = Field(
        "all",
        description='Either "all", a list of ints, or a dict like {"range":"1-100","every":1}.'
    )
    join_key: str = Field(DEFAULT_JOIN_KEY, description="Link DOE to RAW (must be step_number).")
    strict_single_block: bool = Field(
        False, description="Fail if more than one block_id shows up in DOE after filtering."
    )

    @validator("join_key")
    def enforce_join_key(cls, v: str) -> str:
        if v != DEFAULT_JOIN_KEY:
            raise ValueError(f"join_key must be '{DEFAULT_JOIN_KEY}', got '{v}'")
        return v

class KPISection(BaseModel):
    preset: Optional[str] = Field(None, description="Optional preset name.")
    list: List[str] = Field(default_factory=list, description="Explicit KPI names.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Per-KPI parameters (optional).")

class Retentions(BaseModel):
    discharge_capacity_col: str = Field("Q_discharge_Ah")
    energy_col: str = Field("E_discharge_Wh")
    block_col: str = Field("block_id")
    enabled: bool = True

class OutputSpec(BaseModel):
    dir: str
    table_filename: str = Field(DEFAULT_TABLE_FILENAME)
    log_filename: str = Field(DEFAULT_LOG_FILENAME)
    summary_filename: str = Field(DEFAULT_SUMMARY_FILENAME)

class Provenance(BaseModel):
    code_version: Optional[str] = None
    pipeline: Optional[str] = None
    notes: Optional[str] = None

class JobConfig(BaseModel):
    # header
    version: int = Field(JOB_SCHEMA_VERSION)
    created_at: str
    job_id: str

    # core
    cell: str
    selection: Selection
    kpis: KPISection
    retentions: Retentions

    # references
    input_artifacts: Dict[str, str] = Field(
        ..., description="Paths to manifest and optionally raw/doe if desired."
    )
    output: OutputSpec

    # meta
    provenance: Provenance = Field(default_factory=Provenance)

    # ------------------- validators -------------------

    @root_validator
    def validate_kpis(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        kpi_section: KPISection = values.get("kpis")
        preset = (kpi_section.preset or "").strip()
        explicit = list(kpi_section.list or [])

        # Expand preset if provided
        if preset:
            if preset not in KPI_PRESETS:
                raise ValueError(f"Unknown KPI preset: '{preset}'")
            expanded = list(KPI_PRESETS[preset])
        else:
            expanded = []

        # Merge unique names preserving order: preset first, then explicit
        seen = set()
        merged: List[str] = []
        for name in [*expanded, *explicit]:
            if name not in seen:
                seen.add(name)
                merged.append(name)
        kpi_section.list = merged
        values["kpis"] = kpi_section

        if not kpi_section.list:
            raise ValueError("KPI list is empty after preset expansion; specify at least one KPI.")
        return values

    @validator("created_at")
    def validate_created_at(cls, v: str) -> str:
        # ISO-like timestamp check (loose)
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception as e:
            raise ValueError(f"created_at must be ISO-8601 string, got '{v}'") from e
        return v

# ---------------------------------------------------------------------------
# Builder API
# ---------------------------------------------------------------------------

def _sanitize_cell_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _default_job_id(cell: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"{JOB_FILE_PREFIX}{_sanitize_cell_name(cell)}_{ts}_{rand}"

def _resolve_artifacts_root(project_root: Path) -> Path:
    return (project_root / "artifacts").resolve()

def _manifest_for_cell(artifacts_root: Path, cell: str) -> Path:
    return (artifacts_root / _sanitize_cell_name(cell) / "manifest.yaml").resolve()

def _default_output_dir(artifacts_root: Path, cell: str, job_id: str) -> Path:
    return (artifacts_root / _sanitize_cell_name(cell) / "kpis" / job_id).resolve()

def build_job_config(
    *,
    project_root: Path,
    cell: str,
    blocks: List[int],
    cycles: Any = "all",
    kpis: Optional[List[str]] = None,
    kpi_preset: Optional[str] = None,
    retentions_enabled: bool = True,
    discharge_capacity_col: str = "Q_discharge_Ah",
    energy_col: str = "E_discharge_Wh",
    block_col: str = "block_id",
    strict_single_block: bool = False,
    manifest_path: Optional[Path] = None,
    code_version: Optional[str] = None,
    pipeline_ref: Optional[str] = None,
    notes: Optional[str] = None,
    output_dir: Optional[Path] = None,
    job_id: Optional[str] = None,
    kpi_registry: Optional[KPIRegistry] = None,
) -> JobConfig:
    """
    Create a validated JobConfig model. Does not write to disk.
    """
    assert blocks and all(isinstance(b, int) for b in blocks), "blocks must be a non-empty list of ints"

    artifacts_root = _resolve_artifacts_root(Path(project_root))
    manifest = Path(manifest_path) if manifest_path else _manifest_for_cell(artifacts_root, cell)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found for cell '{cell}': {manifest}")

    job_id = job_id or _default_job_id(cell)
    out_dir = Path(output_dir) if output_dir else _default_output_dir(artifacts_root, cell, job_id)

    kpi_list = list(kpis or [])
    if kpi_preset and not kpi_list:
        # OK — will expand from preset in JobConfig root validator
        pass
    elif not kpi_preset and not kpi_list:
        raise ValueError("Provide either `kpi_preset` or a non-empty `kpis` list.")

    # Registry (soft) validation — warns/filters invalid names if provided
    if kpi_registry is not None and kpi_list:
        invalid = [n for n in kpi_list if not kpi_registry.is_valid(n)]
        if invalid:
            raise ValueError(f"Unknown KPI names (registry): {invalid}")

    cfg = JobConfig(
        version=JOB_SCHEMA_VERSION,
        created_at=_now_iso(),
        job_id=job_id,
        cell=cell,
        selection=Selection(
            blocks=blocks,
            cycles=cycles,
            join_key=DEFAULT_JOIN_KEY,
            strict_single_block=strict_single_block,
        ),
        kpis=KPISection(
            preset=kpi_preset,
            list=kpi_list,
            params={},
        ),
        retentions=Retentions(
            discharge_capacity_col=discharge_capacity_col,
            energy_col=energy_col,
            block_col=block_col,
            enabled=retentions_enabled,
        ),
        input_artifacts={
            "manifest": str(manifest),
        },
        output=OutputSpec(
            dir=str(out_dir),
            table_filename=DEFAULT_TABLE_FILENAME,
            log_filename=DEFAULT_LOG_FILENAME,
            summary_filename=DEFAULT_SUMMARY_FILENAME,
        ),
        provenance=Provenance(
            code_version=code_version,
            pipeline=pipeline_ref,
            notes=notes,
        ),
    )
    return cfg

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_job_config(cfg: JobConfig, mkdir: bool = True) -> Path:
    """
    Write YAML to `output.dir`'s sibling jobs folder:
      artifacts/<cell>/jobs/<job_id>.yaml
    Returns the path to the YAML file.
    """
    out_dir = Path(cfg.output.dir).resolve()
    cell_dir = out_dir.parent  # artifacts/<cell>/kpis
    jobs_dir = cell_dir.parent / "jobs"

    if mkdir:
        jobs_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = jobs_dir / f"{cfg.job_id}.yaml"
    # Write as YAML without external deps: emit JSON-compatible YAML
    content = json.loads(cfg.json(by_alias=True, exclude_none=True))
    # Simple YAML emitter to avoid adding PyYAML dependency here
    try:
        import yaml  # type: ignore
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(content, f, sort_keys=False)
    except Exception:
        # Fall back to JSON if YAML not available
        with open(yaml_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)
        return yaml_path.with_suffix(".json")

    return yaml_path

def load_job_config(path: Path) -> JobConfig:
    """
    Load a job file (YAML or JSON) and return a validated JobConfig.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return JobConfig.parse_obj(data)

# ---------------------------------------------------------------------------
# One-liner convenience
# ---------------------------------------------------------------------------

def create_job_file(
    *,
    project_root: Path,
    cell: str,
    blocks: List[int],
    cycles: Any = "all",
    kpis: Optional[List[str]] = None,
    kpi_preset: Optional[str] = None,
    strict_single_block: bool = False,
    code_version: Optional[str] = None,
    pipeline_ref: Optional[str] = None,
    notes: Optional[str] = None,
) -> Path:
    """
    Build → validate → write a job file in one call.
    Returns the path to the created YAML (or JSON if yaml not installed).
    """
    cfg = build_job_config(
        project_root=project_root,
        cell=cell,
        blocks=blocks,
        cycles=cycles,
        kpis=kpis,
        kpi_preset=kpi_preset,
        strict_single_block=strict_single_block,
        code_version=code_version,
        pipeline_ref=pipeline_ref,
        notes=notes,
    )
    return write_job_config(cfg)

# ---------------------------------------------------------------------------
# Minimal CLI (optional)
#   python -m batterydata.kpi_jobs.config_builder --cell CellA --blocks 2 --preset cycle_core
# ---------------------------------------------------------------------------

def _parse_cli_args(argv: List[str]) -> Dict[str, Any]:  # tiny, no deps
    import argparse
    p = argparse.ArgumentParser(description="Build KPI job config.")
    p.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[3]))
    p.add_argument("--cell", required=True)
    p.add_argument("--blocks", required=True, help="Comma-separated block IDs, e.g. 2 or 2,3")
    p.add_argument("--cycles", default="all", help='Use "all", "1-100", or a comma list like 1,2,3')
    p.add_argument("--preset", default=None)
    p.add_argument("--kpis", default=None, help="Comma list of KPI names; ignored if --preset set.")
    p.add_argument("--strict-single-block", action="store_true")
    p.add_argument("--notes", default=None)
    return vars(p.parse_args(argv))

def _normalize_cycles(cycles_arg: str) -> Any:
    s = (cycles_arg or "").strip()
    if s.lower() == "all" or s == "":
        return "all"
    if re.fullmatch(r"\d+\-\d+", s):
        return {"range": s, "every": 1}
    # else try list of ints
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except Exception:
        raise ValueError(f"Unrecognized cycles format: {cycles_arg}")

def _main_cli(argv: List[str]) -> int:
    args = _parse_cli_args(argv)
    project_root = Path(args["project_root"])
    cell = args["cell"]
    blocks = [int(b) for b in str(args["blocks"]).split(",")]
    cycles = _normalize_cycles(args["cycles"])
    preset = args["preset"]
    kpis = None if preset else ([x for x in (args["kpis"] or "").split(",") if x])

    path = create_job_file(
        project_root=project_root,
        cell=cell,
        blocks=blocks,
        cycles=cycles,
        kpi_preset=preset,
        kpis=kpis,
        strict_single_block=args["strict_single_block"],
        notes=args["notes"],
    )
    print(str(path))
    return 0

if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main_cli(sys.argv[1:]))
