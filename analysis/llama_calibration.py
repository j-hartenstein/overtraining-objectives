#!/usr/bin/env python3
"""Reproduce the Llama 3 loss-surface fits and inverse compactness wedges."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


RELEASES = (
    ("Llama-3-8B", 8e9, 15e12),
    ("Llama-3-70B", 70e9, 15e12),
    ("Llama-3-405B", 405e9, 15.6e12),
)
META_FRONTIER_ANCHOR = {"params": 402e9, "tokens": 16.55e12}
META_TOKEN_EXPONENT = 0.53
FLAGSHIP_COMPUTE = 3.8e25
RATE_MULTIPLIERS = (0.75, 1.0, 1.25)


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows = [{key: float(row[key]) for key in ("params", "tokens", "budget", "loss")} for row in raw]
    if len(rows) != 133:
        raise ValueError(f"Expected 133 Llama 3 observations, found {len(rows)}")
    return rows


def fit_meta_constrained(rows: list[dict[str, float]]) -> dict[str, float]:
    """Fit while imposing Meta's frontier exponent and flagship allocation."""
    y = np.array([row["loss"] for row in rows])
    log_n = np.log([row["params"] for row in rows])
    log_d = np.log([row["tokens"] for row in rows])
    ref_n, ref_d = float(np.median(log_n)), float(np.median(log_d))
    n0, d0 = META_FRONTIER_ANCHOR["params"], META_FRONTIER_ANCHOR["tokens"]
    exponent_ratio = (1 - META_TOKEN_EXPONENT) / META_TOKEN_EXPONENT

    def prediction(theta: np.ndarray) -> tuple[np.ndarray, tuple[float, ...]]:
        e, log_b, log_alpha = theta
        alpha = math.exp(log_alpha)
        beta = alpha * exponent_ratio
        b = math.exp(log_b)
        token_at_anchor = b * math.exp(-beta * (math.log(d0) - ref_d))
        parameter_at_anchor = (beta / alpha) * token_at_anchor
        a = parameter_at_anchor / math.exp(-alpha * (math.log(n0) - ref_n))
        fitted = e + a * np.exp(-alpha * (log_n - ref_n)) + b * np.exp(-beta * (log_d - ref_d))
        return fitted, (e, a, b, alpha, beta)

    candidates = []
    for exponent in (0.1, 0.2, 0.3, 0.5, 0.8):
        candidates.append(
            least_squares(
                lambda theta: prediction(theta)[0] - y,
                [0.6, math.log(0.1), math.log(exponent)],
                bounds=([0, -20, math.log(0.02)], [float(y.min()) * 0.999, 20, math.log(2)]),
                max_nfev=50_000,
            )
        )
    fit = min(candidates, key=lambda candidate: float(np.sum(candidate.fun**2)))
    fitted, (e, a, b, alpha, beta) = prediction(fit.x)
    return {
        "E": float(e),
        "A_at_reference_n": float(a),
        "B_at_reference_d": float(b),
        "alpha": float(alpha),
        "beta": float(beta),
        "log_n_reference": ref_n,
        "log_d_reference": ref_d,
        "rmse": float(np.sqrt(np.mean((fitted - y) ** 2))),
        "sse": float(np.sum((fitted - y) ** 2)),
        "converged": bool(fit.success),
    }


def fit_unconstrained(rows: list[dict[str, float]]) -> dict[str, float]:
    y = np.array([row["loss"] for row in rows])
    log_n = np.log([row["params"] for row in rows])
    log_d = np.log([row["tokens"] for row in rows])
    ref_n, ref_d = float(np.median(log_n)), float(np.median(log_d))

    def prediction(theta: np.ndarray) -> np.ndarray:
        e, log_a, log_b, log_alpha, log_beta = theta
        alpha, beta = math.exp(log_alpha), math.exp(log_beta)
        return e + np.exp(log_a - alpha * (log_n - ref_n)) + np.exp(log_b - beta * (log_d - ref_d))

    candidates = []
    for exponent in (0.1, 0.3, 0.7, 1.2):
        for floor_fraction in (0.5, 0.8, 0.95):
            e0 = float(y.min()) * floor_fraction
            amplitude = max(float(np.median(y)) - e0, 0.05) / 2
            candidates.append(
                least_squares(
                    lambda theta: prediction(theta) - y,
                    [e0, math.log(amplitude), math.log(amplitude), math.log(exponent), math.log(exponent)],
                    bounds=(
                        [0, -20, -20, math.log(0.01), math.log(0.01)],
                        [float(y.min()) * 0.999, 20, 20, math.log(2), math.log(2)],
                    ),
                    max_nfev=50_000,
                )
            )
    fit = min(candidates, key=lambda candidate: float(np.sum(candidate.fun**2)))
    e, log_a, log_b, log_alpha, log_beta = fit.x
    fitted = prediction(fit.x)
    return {
        "E": float(e),
        "A_at_reference_n": float(math.exp(log_a)),
        "B_at_reference_d": float(math.exp(log_b)),
        "alpha": float(math.exp(log_alpha)),
        "beta": float(math.exp(log_beta)),
        "log_n_reference": ref_n,
        "log_d_reference": ref_d,
        "rmse": float(np.sqrt(np.mean((fitted - y) ** 2))),
        "sse": float(np.sum((fitted - y) ** 2)),
        "converged": bool(fit.success),
    }


def surface_loss(fit: dict[str, float], n: float, d: float) -> float:
    parameter_term = fit["A_at_reference_n"] * math.exp(
        -fit["alpha"] * (math.log(n) - fit["log_n_reference"])
    )
    token_term = fit["B_at_reference_d"] * math.exp(
        -fit["beta"] * (math.log(d) - fit["log_d_reference"])
    )
    return fit["E"] + parameter_term + token_term


def marginal_rate(fit: dict[str, float], n: float, d: float) -> float:
    parameter_term = fit["A_at_reference_n"] * math.exp(
        -fit["alpha"] * (math.log(n) - fit["log_n_reference"])
    )
    token_term = fit["B_at_reference_d"] * math.exp(
        -fit["beta"] * (math.log(d) - fit["log_d_reference"])
    )
    return (fit["alpha"] * parameter_term / n) / (fit["beta"] * token_term / d)


def compute_optimum(fit: dict[str, float], compute: float = FLAGSHIP_COMPUTE) -> dict[str, float]:
    numerator = (
        math.log(fit["alpha"] * fit["A_at_reference_n"])
        - math.log(fit["beta"] * fit["B_at_reference_d"])
        + fit["alpha"] * fit["log_n_reference"]
        - fit["beta"] * fit["log_d_reference"]
        + fit["beta"] * math.log(compute / 6)
    )
    log_n = numerator / (fit["alpha"] + fit["beta"])
    n = math.exp(log_n)
    return {"params": n, "tokens": compute / (6 * n)}


def total_wedge(fit: dict[str, float], n: float, d: float, delta: float = 0.0) -> float:
    return 3 * (marginal_rate(fit, n, d) * (n + delta) - d)


def common_wedge_test(fit: dict[str, float], r1: float | None = None, r2: float | None = None) -> dict[str, float | bool]:
    _, n1, d1 = RELEASES[0]
    _, n2, d2 = RELEASES[1]
    r1 = marginal_rate(fit, n1, d1) if r1 is None else r1
    r2 = marginal_rate(fit, n2, d2) if r2 is None else r2
    z1, z2 = r1 * n1 - d1, r2 * n2 - d2
    delta = (z2 - z1) / (r1 - r2)
    q_total = 3 * (z1 + r1 * delta)
    return {
        "common_data_wedge": delta,
        "common_parameter_wedge": q_total,
        "nonnegative_data_wedge": bool(delta >= 0),
    }


def flatness(fit: dict[str, float]) -> list[dict[str, float]]:
    optimum = compute_optimum(fit)
    baseline = surface_loss(fit, optimum["params"], optimum["tokens"])
    output = []
    for factor in (0.1, 0.5, 2.0, 10.0):
        loss = surface_loss(fit, optimum["params"] * factor, optimum["tokens"] / factor)
        output.append(
            {
                "parameter_multiplier": factor,
                "token_multiplier": 1 / factor,
                "relative_loss_increase_percent": 100 * (loss - baseline) / baseline,
            }
        )
    return output


def fit_summary(name: str, fit: dict[str, float]) -> dict:
    optimum = compute_optimum(fit)
    releases = []
    for model, n, d in RELEASES:
        q_total = total_wedge(fit, n, d)
        releases.append(
            {
                "model": model,
                "params": n,
                "tokens": d,
                "marginal_rate": marginal_rate(fit, n, d),
                "total_parameter_wedge": q_total,
                "parameter_cost_multiplier": 1 + q_total / (3 * d),
            }
        )
    return {
        "name": name,
        "surface": fit,
        "flagship_compute_optimum": optimum,
        "flatness": flatness(fit),
        "released_endpoints": releases,
        "common_wedge_test_8B_70B": common_wedge_test(fit),
    }


def sensitivity_grid(fit_name: str, fit: dict[str, float]) -> list[dict]:
    _, n8, d8 = RELEASES[0]
    _, n70, d70 = RELEASES[1]
    base_r8 = marginal_rate(fit, n8, d8)
    base_r70 = marginal_rate(fit, n70, d70)
    output = []
    for m8, m70 in itertools.product(RATE_MULTIPLIERS, repeat=2):
        r8, r70 = m8 * base_r8, m70 * base_r70
        common = common_wedge_test(fit, r8, r70)
        output.append(
            {
                "fit": fit_name,
                "rate_multiplier_8B": m8,
                "rate_multiplier_70B": m70,
                "total_wedge_8B": 3 * (r8 * n8 - d8),
                "total_wedge_70B": 3 * (r70 * n70 - d70),
                **common,
            }
        )
    return output


def leave_one_budget_out(rows: list[dict[str, float]]) -> list[dict]:
    output = []
    for budget in sorted({row["budget"] for row in rows}):
        fit = fit_meta_constrained([row for row in rows if row["budget"] != budget])
        output.append(
            {
                "omitted_budget": budget,
                "rmse": fit["rmse"],
                "total_wedge_8B": total_wedge(fit, RELEASES[0][1], RELEASES[0][2]),
                "total_wedge_70B": total_wedge(fit, RELEASES[1][1], RELEASES[1][2]),
            }
        )
    return output


def support_summary(rows: list[dict[str, float]]) -> dict:
    maximum = {
        "params": max(row["params"] for row in rows),
        "tokens": max(row["tokens"] for row in rows),
        "compute": max(row["budget"] for row in rows),
    }
    release_multiples = []
    for model, n, d in RELEASES:
        release_multiples.append(
            {
                "model": model,
                "parameter_multiple": n / maximum["params"],
                "token_multiple": d / maximum["tokens"],
                "compute_multiple": (6 * n * d) / maximum["compute"],
            }
        )
    return {
        "observations": len(rows),
        "compute_budgets": len({row["budget"] for row in rows}),
        "experimental_minimum": {
            "params": min(row["params"] for row in rows),
            "tokens": min(row["tokens"] for row in rows),
            "compute": min(row["budget"] for row in rows),
        },
        "experimental_maximum": maximum,
        "release_to_experimental_maximum": release_multiples,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isoflop-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.isoflop_csv)
    constrained = fit_meta_constrained(rows)
    unconstrained = fit_unconstrained(rows)
    fits = {
        "meta_constrained": fit_summary("meta_constrained", constrained),
        "unconstrained": fit_summary("unconstrained", unconstrained),
    }
    sensitivity = sensitivity_grid("meta_constrained", constrained) + sensitivity_grid("unconstrained", unconstrained)
    lobo = leave_one_budget_out(rows)

    result = {
        "estimand": {
            "total_parameter_wedge": "Q = q + ((C^N)' - B_N)/2, in compute-equivalent generated-token units.",
            "data_wedge": "delta = (C^D)'/6, in parameter-equivalent units.",
            "inverse": "Q(delta) = 3 * (R * (N + delta) - D).",
        },
        "support": support_summary(rows),
        "fits": fits,
        "rate_sensitivity": sensitivity,
        "leave_one_budget_out_meta_constrained": lobo,
        "leave_one_budget_out_ranges": {
            "total_wedge_8B_min": min(row["total_wedge_8B"] for row in lobo),
            "total_wedge_8B_max": max(row["total_wedge_8B"] for row in lobo),
            "total_wedge_70B_min": min(row["total_wedge_70B"] for row in lobo),
            "total_wedge_70B_max": max(row["total_wedge_70B"] for row in lobo),
        },
    }

    table_rows = []
    for fit_name, summary in fits.items():
        releases = {row["model"]: row for row in summary["released_endpoints"]}
        table_rows.append(
            {
                "fit": fit_name,
                "rmse": summary["surface"]["rmse"],
                "optimal_params": summary["flagship_compute_optimum"]["params"],
                "optimal_tokens": summary["flagship_compute_optimum"]["tokens"],
                "total_wedge_8B": releases["Llama-3-8B"]["total_parameter_wedge"],
                "total_wedge_70B": releases["Llama-3-70B"]["total_parameter_wedge"],
                "common_data_wedge": summary["common_wedge_test_8B_70B"]["common_data_wedge"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "llama_calibration.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    write_csv(args.output_dir / "llama_calibration_table.csv", table_rows)
    write_csv(args.output_dir / "local_rate_sensitivity.csv", sensitivity)
    write_csv(args.output_dir / "leave_one_budget_out.csv", lobo)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
