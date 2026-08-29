#!/usr/bin/env python3
"""Verify that generated outputs reproduce the manuscript's numerical claims."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def close(actual: float, expected: float, absolute: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=absolute):
        raise AssertionError(f"Expected {expected} +/- {absolute}, found {actual}")


def main() -> None:
    panel = json.loads((ROOT / "results/panel_estimates.json").read_text())
    estimate = panel["fixed_effect_estimate"]
    assert panel["coverage"]["panel_models"] == 56
    assert panel["coverage"]["developers"] == 5
    assert estimate["models"] == 32
    assert estimate["generation_clusters"] == 11
    close(estimate["slope_log_tokens_on_log_parameters"], 0.2263410455, 1e-9)
    close(estimate["generation_clustered_standard_error"], 0.1295417024, 1e-9)
    close(estimate["tokens_multiplier_for_10x_parameters"], 1.684, 0.002)
    assert panel["all_generation_slopes_below_one"]
    assert panel["all_smallest_siblings_more_token_intensive"]
    close(panel["minimum_smallest_to_largest_contrast"], 1.75, 1e-9)
    close(panel["maximum_smallest_to_largest_contrast"], 48.6778846154, 1e-9)

    calibration = json.loads((ROOT / "results/llama_calibration.json").read_text())
    constrained = calibration["fits"]["meta_constrained"]
    unconstrained = calibration["fits"]["unconstrained"]
    constrained_releases = {row["model"]: row for row in constrained["released_endpoints"]}
    unconstrained_releases = {row["model"]: row for row in unconstrained["released_endpoints"]}

    close(constrained["surface"]["rmse"], 0.00397, 0.000005)
    close(unconstrained["surface"]["rmse"], 0.00390, 0.000005)
    close(constrained["flagship_compute_optimum"]["params"] / 1e9, 393, 1.0)
    close(constrained["flagship_compute_optimum"]["tokens"] / 1e12, 16.1, 0.1)
    close(unconstrained["flagship_compute_optimum"]["params"] / 1e9, 622, 1.0)
    close(unconstrained["flagship_compute_optimum"]["tokens"] / 1e12, 10.2, 0.1)
    close(constrained_releases["Llama-3-8B"]["total_parameter_wedge"] / 1e12, 114.8, 0.1)
    close(constrained_releases["Llama-3-70B"]["total_parameter_wedge"] / 1e12, 33.0, 0.1)
    close(unconstrained_releases["Llama-3-8B"]["total_parameter_wedge"] / 1e12, 150.9, 0.1)
    close(unconstrained_releases["Llama-3-70B"]["total_parameter_wedge"] / 1e12, 55.0, 0.1)
    close(constrained["common_wedge_test_8B_70B"]["common_data_wedge"] / 1e9, -4.34, 0.01)
    close(unconstrained["common_wedge_test_8B_70B"]["common_data_wedge"] / 1e9, -4.16, 0.01)

    sensitivity = calibration["rate_sensitivity"]
    close(min(row["total_wedge_8B"] for row in sensitivity) / 1e12, 74.9, 0.1)
    close(max(row["total_wedge_8B"] for row in sensitivity) / 1e12, 199.8, 0.1)
    close(min(row["total_wedge_70B"] for row in sensitivity) / 1e12, 13.5, 0.1)
    close(max(row["total_wedge_70B"] for row in sensitivity) / 1e12, 80.0, 0.1)
    assert all(row["common_data_wedge"] < 0 for row in sensitivity)

    lobo = calibration["leave_one_budget_out_ranges"]
    close(lobo["total_wedge_8B_min"] / 1e12, 111.1, 0.1)
    close(lobo["total_wedge_8B_max"] / 1e12, 122.0, 0.1)
    close(lobo["total_wedge_70B_min"] / 1e12, 32.2, 0.1)
    close(lobo["total_wedge_70B_max"] / 1e12, 34.5, 0.1)

    figure = ROOT / "figures/allocation_contrasts.pdf"
    assert figure.exists() and figure.stat().st_size > 1_000
    print("PASS: all reported estimates and robustness ranges reproduce.")


if __name__ == "__main__":
    main()
