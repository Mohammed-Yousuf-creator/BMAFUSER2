#!/usr/bin/env python3
"""Build daily Tmax/Tmin model targets from actual forecast timesteps.

The model inputs remain the raw 2-m temperature field produced by the AIFS and
GFS fetchers. IMD daily temperature observations are tied to the observation
date. For temperature verification we use the IMD 24-hour observation window
(0830 IST to 0830 IST of the next day), represented in UTC as 03:00 to 03:00.
Because the current AIFS archive is available on 6-hour forecast increments,
we use the actual model timesteps inside that window: 06, 12, 18 and 00 UTC.
For a target date D, the preferred model run is the 00 UTC initialization on
D-1, giving F006, F012, F018 and F024. No hourly values are invented or
interpolated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from bma_core import LOCATIONS


@dataclass(frozen=True)
class TemperatureExtremeRecord:
    target_date: pd.Timestamp
    location: str
    tmax: float
    tmin: float
    initialization_time: pd.Timestamp
    n_timesteps: int


def _coord_name(ds: xr.Dataset, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
    raise ValueError(f"Missing coordinate among {tuple(candidates)}; found {list(ds.coords)}")


def _data_var(ds: xr.Dataset) -> str:
    for name in ("temperature", "t2m"):
        if name in ds.data_vars:
            return name
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise ValueError(f"Could not identify temperature variable: {list(ds.data_vars)}")


def _as_naive_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(np.asarray(value).reshape(-1)[0])
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def _extract_file(path: Path) -> tuple[pd.Timestamp, dict[str, list[tuple[pd.Timestamp, float]]]]:
    with xr.open_dataset(path) as ds:
        if "forecast_hour" not in ds.coords and "forecast_hour" not in ds.dims:
            raise ValueError("Missing forecast_hour")
        init = _as_naive_timestamp(ds["init_time"].values) if "init_time" in ds.coords else None
        if init is None:
            raise ValueError("Missing init_time")

        if "valid_time" in ds.coords:
            vt_values = np.asarray(ds["valid_time"].values).reshape(-1)
            fh_values = np.asarray(ds["forecast_hour"].values).reshape(-1)
            # With the standardized fetcher schema, valid_time is along forecast_hour.
            if len(vt_values) != len(fh_values):
                raise ValueError("valid_time and forecast_hour lengths do not match")
            valid_times = [_as_naive_timestamp(v) for v in vt_values]
        else:
            fh_values = np.asarray(ds["forecast_hour"].values).reshape(-1)
            valid_times = [init + pd.Timedelta(hours=int(h)) for h in fh_values]

        lat_name = _coord_name(ds, ("latitude", "lat"))
        lon_name = _coord_name(ds, ("longitude", "lon"))
        da = ds[_data_var(ds)]

        out: dict[str, list[tuple[pd.Timestamp, float]]] = {loc: [] for loc in LOCATIONS}
        for i, vt in enumerate(valid_times):
            # Select one forecast hour at a time so 3-D/4-D arrays remain manageable.
            try:
                if "forecast_hour" in da.dims:
                    step = da.isel(forecast_hour=i)
                else:
                    step = da
                for location, (lat, lon) in LOCATIONS.items():
                    grid_lons = np.asarray(ds[lon_name].values, dtype=float)
                    target_lon = lon
                    if grid_lons.size and np.nanmin(grid_lons) >= 0 and np.nanmax(grid_lons) > 180 and lon < 0:
                        target_lon = lon % 360.0
                    value = float(step.sel({lat_name: lat, lon_name: target_lon}, method="nearest").load().values)
                    if np.isfinite(value):
                        out[location].append((vt, value))
            except Exception as exc:
                raise ValueError(f"Cannot extract forecast timestep {vt}: {exc}") from exc
        return init, out


def build_daily_extremes(
    root: Path,
    *,
    start_date: str,
    end_date: str,
    min_timesteps: int = 4,
) -> dict[tuple[str, pd.Timestamp], TemperatureExtremeRecord]:
    """Return the latest usable forecast-run-derived Tmax/Tmin for each date/location."""
    if min_timesteps < 2:
        raise ValueError("min_timesteps must be >= 2")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    files = sorted(root.rglob("*_temperature.nc")) if root.exists() else []
    if not files:
        raise FileNotFoundError(f"No '*_temperature.nc' files under {root}")

    # Candidates are keyed by target date/location. Multiple runs may cover a
    # date; choose the latest initialization strictly before that date.
    candidates: dict[tuple[str, pd.Timestamp], TemperatureExtremeRecord] = {}

    for path in files:
        try:
            init, series = _extract_file(path)
        except Exception as exc:
            print(f"WARNING: skipping temperature file {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        # Group only the dates actually represented by this file. This keeps
        # the operation proportional to the available forecast data rather
        # than looping through the complete requested date range per file.
        for location, points in series.items():
            by_day: dict[pd.Timestamp, dict[pd.Timestamp, float]] = {}
            for vt, value in points:
                # IMD's daily temperature observation window is represented as
                # 03:00 UTC on D-1 through 03:00 UTC on D. We use only actual
                # six-hourly model values falling inside this window.
                candidate_day = vt.normalize() + (pd.Timedelta(days=1) if vt.hour >= 3 else pd.Timedelta(0))
                window_start = candidate_day - pd.Timedelta(hours=21)
                window_end = candidate_day + pd.Timedelta(hours=3)
                if candidate_day < start or candidate_day > end:
                    continue
                if vt < window_start or vt > window_end or init >= window_start:
                    continue
                by_day.setdefault(candidate_day, {})[vt] = value

            for day, values_by_time in by_day.items():
                if len(values_by_time) < min_timesteps:
                    continue
                values = np.asarray(list(values_by_time.values()), dtype=float)
                if not np.all(np.isfinite(values)):
                    continue
                record = TemperatureExtremeRecord(
                    target_date=day,
                    location=location,
                    tmax=float(np.max(values)),
                    tmin=float(np.min(values)),
                    initialization_time=init,
                    n_timesteps=len(values_by_time),
                )
                key = (location, day)
                existing = candidates.get(key)
                if existing is None or init > existing.initialization_time:
                    candidates[key] = record

    return candidates


def records_to_frame(records: dict[tuple[str, pd.Timestamp], TemperatureExtremeRecord]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "target_date": rec.target_date,
            "location": rec.location,
            "tmax": rec.tmax,
            "tmin": rec.tmin,
            "initialization_time": rec.initialization_time,
            "n_timesteps": rec.n_timesteps,
        }
        for rec in records.values()
    ]).sort_values(["target_date", "location"]).reset_index(drop=True)
