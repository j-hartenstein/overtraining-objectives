#!/usr/bin/env python3
"""Run every analysis reported in the paper from packaged public data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(completed.returncode)


def main() -> None:
    python = sys.executable
    run(
        [
            python,
            "analysis/panel_analysis.py",
            "--panel-csv",
            "data/model_family_panel.csv",
            "--output-dir",
            "results",
        ]
    )
    run(
        [
            python,
            "analysis/plot_allocations.py",
            "data/model_family_panel.csv",
            "figures/allocation_contrasts.pdf",
        ]
    )
    run(
        [
            python,
            "analysis/llama_calibration.py",
            "--isoflop-csv",
            "data/llama3_isoflop_experiments.csv",
            "--output-dir",
            "results",
        ]
    )
    print("Reproduced the panel estimate, Figure 1, and Llama 3 calibration.")
    print("Run `python verify.py` to compare the outputs with the manuscript values.")


if __name__ == "__main__":
    main()
