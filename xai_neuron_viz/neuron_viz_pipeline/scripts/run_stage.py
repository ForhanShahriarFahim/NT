# =============================================================================
# FILE: neuron_viz_pipeline/scripts/run_stage.py
#
# Purpose:
#   Single entry point for running one, several, or all pipeline stages in
#   sequence. Each stage is launched as a SUBPROCESS so GPU memory is freed
#   cleanly between stages (important for Stage 3 on ViT) and so each stage
#   keeps its own argparse behaviour.
#
# Stage map (matches the existing file layout):
#
#     --stage 1         -> scripts/stage1_extract.py
#     --stage 2         -> scripts/stage2_rank.py
#     --stage 3         -> scripts/stage3_xai_maps.py
#     --stage 4         -> scripts/stage4_crop.py
#     --stage collage   -> scripts/make_collage.py
#     --stage all       -> 1, 2, 3, 4, collage in that order
#
# Multiple stages in one run:
#     --stage 1,2,3,4
#     --stage 3,4,collage
#
# Overrides are forwarded untouched to every sub-stage. Example:
#
#   python neuron_viz_pipeline/scripts/run_stage.py \
#       --config neuron_viz_pipeline/configs/rn152_ixg.yaml \
#       --stage 3,4,collage \
#       --override rank.top_k=50 \
#       --override neuron.channel_id=15
#
# Behaviour notes:
#   - If any stage exits non-zero, run_stage.py halts (does NOT continue).
#     This avoids "stage 4 ran even though stage 3 failed" silent failures.
#   - Environment variables are inherited from the shell (PYTORCH_ALLOC_CONF,
#     CUDA_VISIBLE_DEVICES, etc. carry through automatically).
#   - The Python interpreter is sys.executable, i.e. the same venv you
#     invoked run_stage.py with. No assumption about which `python` binary
#     is on PATH.
# =============================================================================

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Resolve this script's location so we can find sibling stage scripts.
# ---------------------------------------------------------------------------
_here      = Path(__file__).resolve()
_pkg_root  = _here.parent.parent          # neuron_viz_pipeline/
_scripts_dir = _here.parent               # neuron_viz_pipeline/scripts/
_repo_root = _pkg_root.parent             # project root


# ---------------------------------------------------------------------------
# Stage -> script filename mapping. Single source of truth.
# ---------------------------------------------------------------------------

STAGE_SCRIPTS = {
    "1":       "stage1_extract.py",
    "2":       "stage2_rank.py",
    "3":       "stage3_xai_maps.py",
    "4":       "stage4_crop.py",
    "collage": "make_collage.py",
}

# Order used when --stage all is passed
STAGE_ALL_ORDER = ["1", "2", "3", "4", "collage"]


def resolve_stage_script(stage_key: str) -> Path:
    """
    Map a stage identifier (from --stage) to the absolute path of its script.
    Raises ValueError if the key is unknown.
    """
    stage_key = stage_key.strip().lower()
    if stage_key not in STAGE_SCRIPTS:
        valid = ", ".join(STAGE_SCRIPTS.keys())
        raise ValueError(
            f"Unknown stage '{stage_key}'. Valid stages: {valid}, all"
        )
    script = _scripts_dir / STAGE_SCRIPTS[stage_key]
    if not script.is_file():
        raise FileNotFoundError(
            f"Expected stage script not found:\n  {script}\n"
            f"Did you move or rename scripts/?"
        )
    return script


def parse_stage_list(raw: str) -> list:
    """
    Turn `--stage` argument into an ordered list of stage keys.
    Examples:
        "1"           -> ["1"]
        "3,4,collage" -> ["3", "4", "collage"]
        "all"         -> ["1", "2", "3", "4", "collage"]
    Duplicates are preserved in order (user's choice).
    Whitespace around commas is trimmed.
    """
    raw = raw.strip().lower()
    if raw == "all":
        return list(STAGE_ALL_ORDER)

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError(
            f"--stage '{raw}' parsed to empty list. "
            f"Valid examples: 1 | 3,4 | all | 3,4,collage"
        )

    # Validate each one before returning so we fail fast
    for p in parts:
        if p not in STAGE_SCRIPTS:
            valid = ", ".join(STAGE_SCRIPTS.keys())
            raise ValueError(
                f"Unknown stage '{p}' in --stage '{raw}'. "
                f"Valid stages: {valid}, all"
            )
    return parts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Multi-stage dispatcher for neuron_viz_pipeline. Runs each stage in a "
            "subprocess so GPU memory is freed cleanly between stages."
        )
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file, e.g. neuron_viz_pipeline/configs/rn152_ixg.yaml",
    )
    parser.add_argument(
        "--stage", type=str, required=True,
        help=(
            "Which stage(s) to run. Examples:\n"
            "  1             - just Stage 1 (extract activations)\n"
            "  3,4           - Stage 3 then Stage 4\n"
            "  3,4,collage   - Stage 3, 4, then collage\n"
            "  all           - 1, 2, 3, 4, collage\n"
            "  collage       - just the collage step"
        ),
    )
    parser.add_argument(
        "--override", action="append", default=[],
        help=(
            "Override a config value; forwarded to every sub-stage untouched. "
            "Repeat for multiple. Example: --override rank.top_k=50"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Print the stage commands that would be run, but don't execute. "
            "Useful to sanity-check the plan."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Subprocess launcher
# ---------------------------------------------------------------------------

def build_subprocess_cmd(script_path: Path, config_path: str, overrides: list) -> list:
    """
    Build argv for one sub-stage invocation.
    Uses sys.executable to ensure we invoke with the same Python/venv.
    """
    cmd = [sys.executable, str(script_path), "--config", config_path]
    for ov in overrides:
        cmd += ["--override", ov]
    return cmd


def run_one_stage(stage_key: str, config_path: str, overrides: list,
                  dry_run: bool = False) -> int:
    """
    Run a single stage as a subprocess. Returns the subprocess return code.
    Stdout/stderr are inherited (streamed directly to the user's terminal),
    which is what Fahim expects when watching long-running jobs.
    """
    script = resolve_stage_script(stage_key)
    cmd    = build_subprocess_cmd(script, config_path, overrides)

    print("\n" + "-" * 70)
    print(f"STAGE: {stage_key.upper()}  ({STAGE_SCRIPTS[stage_key]})")
    print("-" * 70)
    print("  cmd:", " ".join(cmd))

    if dry_run:
        print("  (dry-run - not executed)")
        return 0

    t0 = time.time()
    # inherit stdout/stderr so user sees live output
    proc = subprocess.run(cmd, cwd=str(_repo_root))
    dt = time.time() - t0

    print(f"\n  [stage {stage_key}] exit={proc.returncode}  elapsed={dt:.1f}s")
    return proc.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Parse and validate the stage list up front.
    try:
        stages = parse_stage_list(args.stage)
    except (ValueError, FileNotFoundError) as e:
        print(f"\nERROR: {e}\n")
        sys.exit(2)

    # Validate config exists (fail fast)
    if not Path(args.config).is_file():
        print(f"\nERROR: config file not found: {args.config}\n")
        sys.exit(2)

    # Print run plan
    print("\n" + "=" * 70)
    print("RUN PLAN")
    print("=" * 70)
    print(f"  config     : {args.config}")
    print(f"  stages     : {' -> '.join(s.upper() for s in stages)}")
    print(f"  overrides  : {args.override if args.override else '(none)'}")
    print(f"  python     : {sys.executable}")
    print(f"  cwd        : {_repo_root}")
    if args.dry_run:
        print(f"  dry-run    : YES - no subprocess will be launched")

    # Walk the plan, halt on first failure
    overall_start = time.time()
    failed_stage  = None
    for stage_key in stages:
        rc = run_one_stage(
            stage_key  = stage_key,
            config_path= args.config,
            overrides  = args.override,
            dry_run    = args.dry_run,
        )
        if rc != 0:
            failed_stage = stage_key
            break

    total_elapsed = time.time() - overall_start

    print("\n" + "=" * 70)
    if failed_stage is not None:
        print(f"HALTED at stage {failed_stage.upper()} (non-zero exit).")
        print(f"Remaining stages skipped.")
        print(f"Total elapsed: {total_elapsed:.1f}s")
        print("=" * 70)
        sys.exit(1)
    else:
        print(f"ALL STAGES COMPLETED")
        print(f"Total elapsed: {total_elapsed:.1f}s")
        print("=" * 70)


if __name__ == "__main__":
    main()
