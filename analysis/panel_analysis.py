#!/usr/bin/env python3
"""Reproduce the within-generation allocation estimates in the paper."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def read_panel(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["total_params_b"] = float(row["total_params_b"])
        row["active_params_b"] = float(row["active_params_b"])
        row["training_tokens_t"] = float(row["training_tokens_t"])
        row["eligible_for_dn"] = parse_bool(row["eligible_for_dn"])
    return rows


def is_dense(row: dict) -> bool:
    architecture = row["architecture"].lower()
    return "moe" not in architecture and "mixture-of-experts" not in architecture


def analyzed_rows(rows: list[dict]) -> tuple[list[dict], list[tuple[str, str]]]:
    eligible = [row for row in rows if row["eligible_for_dn"] and is_dense(row)]
    counts = Counter((row["developer"], row["generation"]) for row in eligible)
    groups = sorted(group for group, count in counts.items() if count >= 2)
    selected = [row for row in eligible if (row["developer"], row["generation"]) in groups]
    return selected, groups


def fixed_effect_estimate(rows: list[dict], groups: list[tuple[str, str]]) -> dict:
    keys = [(row["developer"], row["generation"]) for row in rows]
    x = np.log(np.array([row["total_params_b"] for row in rows], dtype=float))
    y = np.log(np.array([row["training_tokens_t"] for row in rows], dtype=float))

    x_within = np.empty_like(x)
    y_within = np.empty_like(y)
    for group in groups:
        index = np.array([key == group for key in keys])
        x_within[index] = x[index] - x[index].mean()
        y_within[index] = y[index] - y[index].mean()

    beta = float(x_within @ y_within / (x_within @ x_within))
    residual = y_within - beta * x_within
    score_squares = 0.0
    for group in groups:
        index = np.array([key == group for key in keys])
        score_squares += float(x_within[index] @ residual[index]) ** 2

    n = len(rows)
    cluster_count = len(groups)
    parameter_count = cluster_count + 1  # generation effects plus the common slope
    correction = (cluster_count / (cluster_count - 1)) * ((n - 1) / (n - parameter_count))
    variance = correction * score_squares / float(x_within @ x_within) ** 2
    return {
        "slope_log_tokens_on_log_parameters": beta,
        "generation_clustered_standard_error": math.sqrt(variance),
        "models": n,
        "generation_clusters": cluster_count,
        "tokens_multiplier_for_10x_parameters": 10**beta,
        "cluster_correction": "G/(G-1) * (n-1)/(n-G-1)",
    }


def generation_statistics(rows: list[dict], groups: list[tuple[str, str]]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["developer"], row["generation"])].append(row)

    output = []
    for developer, generation in groups:
        values = sorted(grouped[(developer, generation)], key=lambda row: row["total_params_b"])
        x = np.log(np.array([row["total_params_b"] for row in values]))
        y = np.log(np.array([row["training_tokens_t"] for row in values]))
        slope = float(((x - x.mean()) @ (y - y.mean())) / ((x - x.mean()) @ (x - x.mean())))
        smallest_intensity = 1000 * values[0]["training_tokens_t"] / values[0]["total_params_b"]
        largest_intensity = 1000 * values[-1]["training_tokens_t"] / values[-1]["total_params_b"]
        output.append(
            {
                "developer": developer,
                "generation": generation,
                "models": len(values),
                "smallest_params_b": values[0]["total_params_b"],
                "largest_params_b": values[-1]["total_params_b"],
                "smallest_tokens_per_parameter": smallest_intensity,
                "largest_tokens_per_parameter": largest_intensity,
                "smallest_to_largest_token_intensity": smallest_intensity / largest_intensity,
                "generation_slope": slope,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    all_rows = read_panel(args.panel_csv)
    rows, groups = analyzed_rows(all_rows)
    estimates = fixed_effect_estimate(rows, groups)
    by_generation = generation_statistics(rows, groups)
    dense_eligible = [row for row in all_rows if row["eligible_for_dn"] and is_dense(row)]

    result = {
        "coverage": {
            "panel_models": len(all_rows),
            "developers": len({row["developer"] for row in all_rows}),
            "dense_models_with_model_specific_tokens": len(dense_eligible),
            "regression_models": len(rows),
            "multi_model_generations": len(groups),
        },
        "fixed_effect_estimate": estimates,
        "all_generation_slopes_below_one": all(row["generation_slope"] < 1 for row in by_generation),
        "all_smallest_siblings_more_token_intensive": all(
            row["smallest_to_largest_token_intensity"] > 1 for row in by_generation
        ),
        "minimum_smallest_to_largest_contrast": min(
            row["smallest_to_largest_token_intensity"] for row in by_generation
        ),
        "maximum_smallest_to_largest_contrast": max(
            row["smallest_to_largest_token_intensity"] for row in by_generation
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "panel_estimates.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    write_csv(args.output_dir / "within_generation_estimates.csv", by_generation)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
