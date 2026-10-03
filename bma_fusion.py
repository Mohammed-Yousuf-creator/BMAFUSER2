#!/usr/bin/env python3
"""BMA inference for ClimaFuse.

The public variable ``temperature`` is produced from two internally trained
models: Tmax and Tmin. Their predictive distributions are sampled and averaged
under a conditional-independence approximation, producing one daily-mean
probability distribution for the frontend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from bma_core import LOCATIONS, MODEL_VARIABLES, distribution_2d, load_models, sample_mixture


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def fuse_component_distribution(*, variable: str, location: str, aifs_value: float, gfs_value: float,
                                 model_file: str | Path, bins: int = 80) -> list[list[float]]:
    models = load_models(model_file)
    if variable not in models:
        raise KeyError(f"Variable {variable!r} is not present in {model_file}")
    if location not in models[variable]:
        raise KeyError(f"Location {location!r} is not present in {model_file}")
    return distribution_2d(
        models[variable][location], float(aifs_value), float(gfs_value), bins=bins
    )


def fuse_temperature_distribution(*, location: str,
                                   aifs_tmax: float, gfs_tmax: float,
                                   aifs_tmin: float, gfs_tmin: float,
                                   model_file: str | Path, bins: int = 80,
                                   samples: int = 60000) -> list[list[float]]:
    models = load_models(model_file)
    for required in ("tmax", "tmin"):
        if required not in models or location not in models[required]:
            raise KeyError(f"Model artifact must contain {required}/{location}")

    # This is the temperature output: average of separately predicted Tmax/Tmin.
    # We retain a probability distribution rather than just averaging two point
    # forecasts. Conditional independence between the two fitted distributions is
    # the explicit prototype assumption.
    seed = _stable_seed(location, "temperature")
    ss = np.random.SeedSequence(seed)
    child1, child2 = ss.spawn(2)
    seed1 = int(child1.generate_state(1)[0])
    seed2 = int(child2.generate_state(1)[0])
    tmax_samples = sample_mixture(models["tmax"][location], aifs_tmax, gfs_tmax, samples, seed1)
    tmin_samples = sample_mixture(models["tmin"][location], aifs_tmin, gfs_tmin, samples, seed2)
    mean_samples = 0.5 * (tmax_samples + tmin_samples)

    lo, hi = np.quantile(mean_samples, [1e-5, 1.0 - 1e-5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise RuntimeError(f"Degenerate temperature distribution for {location}")
    edges = np.linspace(float(lo), float(hi), bins + 1)
    mass, _ = np.histogram(mean_samples, bins=edges)
    mass = mass.astype(float)
    mass /= mass.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    return [[float(v), float(p)] for v, p in zip(centers, mass, strict=True)]


def fuse_distribution(*, variable: str, location: str, model_file: str | Path,
                      aifs_value: float | None = None, gfs_value: float | None = None,
                      bins: int = 80, **kwargs) -> list[list[float]]:
    """Backward-compatible dispatcher.

    For temperature, supply aifs_tmax/gfs_tmax/aifs_tmin/gfs_tmin.
    For precipitation, supply aifs_value/gfs_value.
    """
    if variable == "temperature":
        return fuse_temperature_distribution(
            location=location,
            aifs_tmax=float(kwargs["aifs_tmax"]),
            gfs_tmax=float(kwargs["gfs_tmax"]),
            aifs_tmin=float(kwargs["aifs_tmin"]),
            gfs_tmin=float(kwargs["gfs_tmin"]),
            model_file=model_file,
            bins=bins,
        )
    if variable not in MODEL_VARIABLES:
        raise ValueError(f"Supported public variables: temperature, precipitation")
    return fuse_component_distribution(
        variable=variable,
        location=location,
        aifs_value=float(aifs_value),
        gfs_value=float(gfs_value),
        model_file=model_file,
        bins=bins,
    )


def build_parser():
    p = argparse.ArgumentParser(description="ClimaFuse BMA probability distribution generator")
    p.add_argument("--variable", choices=("temperature", "precipitation"), required=True)
    p.add_argument("--location", choices=tuple(LOCATIONS), required=True)
    p.add_argument("--model-file", default="./models/bma_models_2025.json")
    p.add_argument("--bins", type=int, default=80)
    p.add_argument("--aifs", type=float, help="AIFS precipitation F24 value")
    p.add_argument("--gfs", type=float, help="GFS precipitation F24 value")
    p.add_argument("--aifs-tmax", type=float)
    p.add_argument("--gfs-tmax", type=float)
    p.add_argument("--aifs-tmin", type=float)
    p.add_argument("--gfs-tmin", type=float)
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.variable == "temperature":
            required = (args.aifs_tmax, args.gfs_tmax, args.aifs_tmin, args.gfs_tmin)
            if any(v is None for v in required):
                raise ValueError("temperature requires --aifs-tmax --gfs-tmax --aifs-tmin --gfs-tmin")
            distribution = fuse_temperature_distribution(
                location=args.location,
                aifs_tmax=args.aifs_tmax,
                gfs_tmax=args.gfs_tmax,
                aifs_tmin=args.aifs_tmin,
                gfs_tmin=args.gfs_tmin,
                model_file=args.model_file,
                bins=args.bins,
            )
        else:
            if args.aifs is None or args.gfs is None:
                raise ValueError("precipitation requires --aifs and --gfs")
            distribution = fuse_component_distribution(
                variable="precipitation", location=args.location,
                aifs_value=args.aifs, gfs_value=args.gfs,
                model_file=args.model_file, bins=args.bins,
            )
        print(json.dumps({
            "forecast_hour": 24,
            "variable": args.variable,
            "location": args.location,
            "distribution": distribution,
        }, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
