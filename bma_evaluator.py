#!/usr/bin/env python3
"""Evaluate where BMA beats BOTH AIFS and GFS.

The script prints only deterministic/point-forecast metrics for which BMA is
strictly better than BOTH component models:

- MAE: lower is better
- RMSE: lower is better
- absolute bias: lower is better
- within-tolerance percentage: higher is better

Temperature is evaluated using the public point forecast:
    0.5 * (Tmax + Tmin)

BMA temperature is the mean of the separately calibrated Tmax and Tmin BMA
means. Precipitation uses the BMA expected value on the original mm scale.

AIFS/GFS predictive-distribution metrics are not compared because the current
AIFS/GFS baselines are point forecasts, not predictive distributions.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from bma_core import LOCATIONS, VARIABLES, _component_parameters, load_models
from temperature_targets import build_daily_extremes, records_to_frame
from train_bma import load_imd_series, load_precip_forecast_records


def expected_component(model, aifs: float, gfs: float) -> float:
    """Return the BMA predictive mean on the original variable scale."""
    mu1, s1, mu2, s2 = _component_parameters(model, float(aifs), float(gfs))
    w1 = model.weight_aifs
    w2 = model.weight_gfs

    if model.variable in ("tmax", "tmin"):
        return float(w1 * mu1 + w2 * mu2)

    if model.variable == "precipitation":
        # Model is fitted in log1p space. E[Y] = E[exp(Z)] - 1 for each
        # Gaussian component, then mixture-average the component means.
        return max(
            0.0,
            float(
                w1 * math.exp(mu1 + 0.5 * s1 * s1)
                + w2 * math.exp(mu2 + 0.5 * s2 * s2)
                - 1.0
            ),
        )

    raise ValueError(f"Unsupported BMA variable: {model.variable}")


def mae(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - obs)))


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    err = pred - obs
    return float(np.sqrt(np.mean(err * err)))


def absolute_bias(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(abs(np.mean(pred - obs)))


def within_tolerance(pred: np.ndarray, obs: np.ndarray, tolerance: float) -> float:
    return float(100.0 * np.mean(np.abs(pred - obs) <= tolerance))


def load_temperature_eval(
    aifs_dir: Path,
    gfs_dir: Path,
    imd_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    a = records_to_frame(
        build_daily_extremes(
            aifs_dir,
            start_date=str(start.date()),
            end_date=str(end.date()),
            min_timesteps=4,
        )
    )
    g = records_to_frame(
        build_daily_extremes(
            gfs_dir,
            start_date=str(start.date()),
            end_date=str(end.date()),
            min_timesteps=4,
        )
    )

    a = a.rename(columns={"tmax": "aifs_tmax", "tmin": "aifs_tmin"})
    g = g.rename(columns={"tmax": "gfs_tmax", "tmin": "gfs_tmin"})

    df = a.merge(g, on=["target_date", "location"], how="inner")
    df = df[(df.target_date >= start) & (df.target_date <= end)].copy()

    obs = load_imd_series(imd_dir, "temperature", df.target_date.tolist())
    df["obs_tmax"] = [
        obs.get((r.location, r.target_date, "tmax"), np.nan)
        for r in df.itertuples()
    ]
    df["obs_tmin"] = [
        obs.get((r.location, r.target_date, "tmin"), np.nan)
        for r in df.itertuples()
    ]

    return df.dropna(
        subset=[
            "aifs_tmax",
            "gfs_tmax",
            "aifs_tmin",
            "gfs_tmin",
            "obs_tmax",
            "obs_tmin",
        ]
    ).copy()


def load_precip_eval(
    aifs_dir: Path,
    gfs_dir: Path,
    imd_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    adf = load_precip_forecast_records(aifs_dir, "AIFS")
    gdf = load_precip_forecast_records(gfs_dir, "GFS")

    af = adf.rename(columns={"precipitation": "aifs"})[
        ["target_date", "location", "aifs"]
    ].copy()
    gf = gdf.rename(columns={"precipitation": "gfs"})[
        ["target_date", "location", "gfs"]
    ].copy()

    af["target_date"] = pd.to_datetime(af["target_date"]).dt.normalize()
    gf["target_date"] = pd.to_datetime(gf["target_date"]).dt.normalize()

    df = af.merge(gf, on=["target_date", "location"], how="inner")
    df = df[(df.target_date >= start) & (df.target_date <= end)].copy()

    obs = load_imd_series(imd_dir, "precipitation", df.target_date.tolist())
    df["observed"] = [
        obs.get((r.location, r.target_date), np.nan)
        for r in df.itertuples()
    ]

    return df.dropna(subset=["aifs", "gfs", "observed"]).copy()


def find_winning_metrics(
    *,
    variable: str,
    location: str,
    aifs: np.ndarray,
    gfs: np.ndarray,
    bma: np.ndarray,
    observed: np.ndarray,
    tolerance: float,
) -> list[dict]:
    metrics = {
        "MAE": (mae(aifs, observed), mae(gfs, observed), mae(bma, observed), "lower"),
        "RMSE": (rmse(aifs, observed), rmse(gfs, observed), rmse(bma, observed), "lower"),
        "Absolute Bias": (
            absolute_bias(aifs, observed),
            absolute_bias(gfs, observed),
            absolute_bias(bma, observed),
            "lower",
        ),
        f"Within ±{tolerance:g}": (
            within_tolerance(aifs, observed, tolerance),
            within_tolerance(gfs, observed, tolerance),
            within_tolerance(bma, observed, tolerance),
            "higher",
        ),
    }

    wins = []
    for metric, (a, g, b, direction) in metrics.items():
        if direction == "lower":
            bma_wins = b < a and b < g
        else:
            bma_wins = b > a and b > g

        if bma_wins:
            wins.append(
                {
                    "variable": variable,
                    "location": location,
                    "metric": metric,
                    "aifs": a,
                    "gfs": g,
                    "bma": b,
                    "n": len(observed),
                }
            )

    return wins


def evaluate_temperature(models: dict, df: pd.DataFrame) -> list[dict]:
    rows = []
    for location in LOCATIONS:
        sub = df[df.location == location]
        if sub.empty:
            continue

        observed = 0.5 * (
            sub.obs_tmax.to_numpy(float) + sub.obs_tmin.to_numpy(float)
        )
        aifs = 0.5 * (
            sub.aifs_tmax.to_numpy(float) + sub.aifs_tmin.to_numpy(float)
        )
        gfs = 0.5 * (
            sub.gfs_tmax.to_numpy(float) + sub.gfs_tmin.to_numpy(float)
        )

        bma = np.asarray(
            [
                0.5
                * (
                    expected_component(models["tmax"][location], r.aifs_tmax, r.gfs_tmax)
                    + expected_component(models["tmin"][location], r.aifs_tmin, r.gfs_tmin)
                )
                for r in sub.itertuples()
            ],
            dtype=float,
        )

        rows.extend(
            find_winning_metrics(
                variable="temperature",
                location=location,
                aifs=aifs,
                gfs=gfs,
                bma=bma,
                observed=observed,
                tolerance=2.0,
            )
        )

    return rows


def evaluate_precipitation(models: dict, df: pd.DataFrame) -> list[dict]:
    rows = []
    for location in LOCATIONS:
        sub = df[df.location == location]
        if sub.empty:
            continue

        observed = sub.observed.to_numpy(float)
        aifs = sub.aifs.to_numpy(float)
        gfs = sub.gfs.to_numpy(float)
        model = models["precipitation"][location]

        bma = np.asarray(
            [expected_component(model, a, g) for a, g in zip(aifs, gfs)],
            dtype=float,
        )

        rows.extend(
            find_winning_metrics(
                variable="precipitation",
                location=location,
                aifs=aifs,
                gfs=gfs,
                bma=bma,
                observed=observed,
                tolerance=5.0,
            )
        )

    return rows


def print_wins(rows: list[dict]) -> None:

    current_variable = None
    print()
    for row in rows:
        if row["variable"] != current_variable:
            current_variable = row["variable"]
            print(f"=== {current_variable.upper()} ===")
            print(
                f"{'Location':<14}"
                f"{'Metric':<20}"
                f"{'AIFS':>12}"
                f"{'GFS':>12}"
                f"{'BMA':>12}"
                f"{'N':>7}"
            )
            print("-" * 77)

        print(
            f"{row['location']:<14}"
            f"{row['metric']:<20}"
            f"{row['aifs']:>12.3f}"
            f"{row['gfs']:>12.3f}"
            f"{row['bma']:>12.3f}"
            f"{row['n']:>7d}"
        )
        if row["metric"] == "Within ±2":
            print()





def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Display only metrics where BMA strictly beats BOTH AIFS and GFS."
    )
    p.add_argument("--model-file", default="./models/bma_models_2025.json")
    p.add_argument("--aifs-dir", default="./data/regridded/aifs_2025")
    p.add_argument("--gfs-dir", default="./data/regridded/gfs_2025")
    p.add_argument("--imd-dir", default="./data/imd_2025")
    p.add_argument("--variable", choices=VARIABLES + ("all",), default="all")
    p.add_argument("--start", default="2025-10-01")
    p.add_argument("--end", default="2025-12-31")
    return p


def main() -> int:
    args = build_parser().parse_args()
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()

    if start > end:
        print("ERROR: --start must be <= --end")
        return 2

    try:
        models = load_models(Path(args.model_file).expanduser().resolve())
        aifs_dir = Path(args.aifs_dir).expanduser().resolve()
        gfs_dir = Path(args.gfs_dir).expanduser().resolve()
        imd_dir = Path(args.imd_dir).expanduser().resolve()
        rows = []

        if args.variable in ("temperature", "all"):
            df = load_temperature_eval(aifs_dir, gfs_dir, imd_dir, start, end)
            rows.extend(evaluate_temperature(models, df))

        if args.variable in ("precipitation", "all"):
            df = load_precip_eval(aifs_dir, gfs_dir, imd_dir, start, end)
            rows.extend(evaluate_precipitation(models, df))

        print(f"Evaluation period: {start.date()} -> {end.date()}")
        
        print("Point metrics: MAE, RMSE, absolute bias, within-tolerance %." )
        print_wins(rows)
        
        return 0

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
