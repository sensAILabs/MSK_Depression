"""
Master runner for the final Mental to MSK analysis pipeline.

This script runs the project modules in the final analysis order.
It does not create its own results, logs, tables, or figures.

Run from the repository root:

    python -m mental_msk_prediction.run_analysis

"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


PACKAGE_NAME = "mental_msk_prediction"

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent


PIPELINE = [
    {
        "key": "clean_data",
        "label": "Clean data",
        "module": f"{PACKAGE_NAME}.preprocessing.clean_data",
    },
    {
        "key": "build_cohort",
        "label": "Build cohort",
        "module": f"{PACKAGE_NAME}.cohort.build_cohort",
    },
    {
        "key": "prepare_atc_definition",
        "label": "Prepare ATC definitions",
        "module": f"{PACKAGE_NAME}.preprocessing.prepare_atc_definition",
    },
    {
        "key": "build_analysis_df",
        "label": "Build analysis dataframe",
        "module": f"{PACKAGE_NAME}.features.build_analysis_df",
    },
    {
        "key": "cohort_eda",
        "label": "Cohort EDA",
        "module": f"{PACKAGE_NAME}.visualization.cohort_eda",
    },
    {
        "key": "ml_modeling",
        "label": "ML modeling",
        "module": f"{PACKAGE_NAME}.modeling.ml_modeling",
    },
    {
        "key": "sensitivity_analysis",
        "label": "Sensitivity analysis",
        "module": f"{PACKAGE_NAME}.modeling.sensitivity_analysis",
    },
    {
        "key": "cluster_analysis",
        "label": "Cluster analysis",
        "module": f"{PACKAGE_NAME}.modeling.cluster_analysis",
    },
    {
        "key": "patient_trajectory",
        "label": "Patient trajectory",
        "module": f"{PACKAGE_NAME}.visualization.patient_trajectory",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final Mental -> MSK analysis pipeline."
    )

    choices = [
        step["key"]
        for step in PIPELINE
    ]

    parser.add_argument(
        "--start-at",
        choices=choices,
        default=None,
        help="Start at this stage and run all later stages.",
    )

    parser.add_argument(
        "--only",
        nargs="+",
        choices=choices,
        default=None,
        help="Run only the listed stages, preserving pipeline order.",
    )

    parser.add_argument(
        "--skip",
        nargs="+",
        choices=choices,
        default=None,
        help="Skip the listed stages.",
    )

    return parser.parse_args()


def module_to_file(
    module: str,
) -> Path:
    parts = module.split(".")

    if (
        not parts
        or parts[0] != PACKAGE_NAME
    ):
        raise ValueError(
            f"Unexpected module outside {PACKAGE_NAME}: {module}"
        )

    return (
        PACKAGE_DIR
        / Path(*parts[1:]).with_suffix(".py")
    )


def validate_pipeline_modules() -> None:
    missing = []

    for step in PIPELINE:
        path = module_to_file(
            step["module"]
        )

        if not path.is_file():
            missing.append(path)

    if missing:
        raise FileNotFoundError(
            "Missing pipeline module(s):\n"
            + "\n".join(
                f"  - {path}"
                for path in missing
            )
        )


def select_steps(
    args: argparse.Namespace,
) -> list[dict]:
    steps = list(
        PIPELINE
    )

    if args.start_at is not None:
        start_index = next(
            index
            for index, step in enumerate(steps)
            if step["key"] == args.start_at
        )

        steps = steps[
            start_index:
        ]

    if args.only:
        only = set(
            args.only
        )

        steps = [
            step
            for step in steps
            if step["key"] in only
        ]

    if args.skip:
        skip = set(
            args.skip
        )

        steps = [
            step
            for step in steps
            if step["key"] not in skip
        ]

    if not steps:
        raise ValueError(
            "No pipeline stages remain after applying the requested options."
        )

    return steps


def run_stage(
    step: dict,
    index: int,
    total: int,
) -> None:
    label = step["label"]
    module = step["module"]

    print(
        f"[{index}/{total}] "
        f"{label:<28}",
        end="",
        flush=True,
    )

    started = time.perf_counter()

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
        ],
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    if completed.returncode != 0:
        print(
            f"✗  {elapsed:7.1f}s"
        )

        raise RuntimeError(
            f"{label} failed "
            f"(return code {completed.returncode})."
        )

    print(
        f"✓  {elapsed:7.1f}s"
    )


def main() -> int:
    args = parse_args()

    try:
        validate_pipeline_modules()

        steps = select_steps(
            args
        )

    except Exception as exc:
        print(
            f"Pipeline configuration error: {exc}",
            file=sys.stderr,
        )
        return 2

    print(
        "=" * 72
    )
    print(
        "MENTAL -> MSK ANALYSIS PIPELINE"
    )
    print(
        "=" * 72
    )

    started = time.perf_counter()

    try:
        for index, step in enumerate(
            steps,
            start=1,
        ):
            run_stage(
                step=step,
                index=index,
                total=len(steps),
            )

    except Exception as exc:
        print(
            "-" * 72
        )
        print(
            f"Pipeline stopped: {exc}"
        )
        print(
            "=" * 72
        )
        return 1

    elapsed = (
        time.perf_counter()
        - started
    )

    print(
        "-" * 72
    )
    print(
        f"Pipeline completed successfully in {elapsed:.1f}s."
    )
    print(
        "=" * 72
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
