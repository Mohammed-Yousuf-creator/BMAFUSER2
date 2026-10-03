#!/usr/bin/env python3
"""Real-time ClimaFuse orchestration layer.

Public inputs:
    temperature
    precipitation

Temperature is trained internally as two separate probabilistic targets:
    tmax and tmin
The public result is ONE temperature probability distribution obtained by
sampling the two predicted distributions and averaging Tmax/Tmin.

For temperature, the orchestrator fetches actual model temperature timesteps
covering the next UTC day and derives the daily extrema. It never fabricates
hourly values or retrains the model at request time.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from bma_core import FORECAST_HOUR, LOCATIONS, VARIABLES, load_models
from bma_fusion import fuse_component_distribution, fuse_temperature_distribution
from train_bma import load_precip_forecast_records
from temperature_targets import build_daily_extremes

DEFAULT_MODEL = "./models/bma_models_2025.json"
DEFAULT_AIFS_FETCHER = "./ecmwf_aifs_fetcher.py"
DEFAULT_GFS_FETCHER = "./noaa_gfs_fetcher.py"
DEFAULT_AIFS_CACHE = "./data/realtime/aifs"
DEFAULT_GFS_CACHE = "./data/realtime/gfs"
CYCLES = (0, 6, 12, 18)


def _load_module(path: Path, name: str):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Fetcher not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _required_temp_fxx(init_hour: int) -> tuple[int, ...]:
    """Actual six-hourly leads used for the IMD daily Tmax/Tmin window.

    The real-time temperature target uses the 00 UTC model cycle from the
    previous date and F006/F012/F018/F024, covering the available model points
    inside the 03:00-03:00 UTC IMD observation window.
    """
    if init_hour != 0:
        raise ValueError("Daily temperature target currently requires the 00 UTC cycle.")
    return (6, 12, 18, 24)


def _candidate_inits(now_utc: datetime, lag_hours: float = 3.0, lookback_hours: int = 30, allowed_hours=CYCLES):
    cutoff = now_utc - timedelta(hours=lag_hours)
    floor = cutoff - timedelta(hours=lookback_hours)
    out = []
    day = cutoff.replace(minute=0, second=0, microsecond=0)
    while day >= floor:
        if day.hour in allowed_hours:
            out.append(day)
        day -= timedelta(hours=1)
    return sorted(set(out), reverse=True)


def _fetch_one_init(variable, init, aifs_mod, gfs_mod, aifs_dir, gfs_dir):
    start = end = init.replace(tzinfo=None)
    if variable == "temperature":
        fxx = _required_temp_fxx(init.hour)
        gfs_fxx = fxx
    else:
        fxx = (24,)
        gfs_fxx = (24,)

    aifs_mod.fetch_aifs(
        start, end, variable if variable != "temperature" else "temperature",
        cycles=(init.hour,),
        fxx=fxx,
        priority=["azure", "aws", "ecmwf", "google"],
        save_dir=aifs_dir,
        overwrite=False,
        fail_fast=False,
        retries=4,
        pause=3.0,
        cooldown_min=10.0,
        max_cooldowns=3,
        max_consecutive_failures=3,
        debug=False,
    )
    gfs_mod.fetch_gfs(
        start, end, variable if variable != "temperature" else "temperature",
        cycles=(init.hour,),
        fxx=gfs_fxx,
        priority=["aws", "google", "nomads", "azure"],
        save_dir=gfs_dir,
        overwrite=False,
        fail_fast=False,
    )
    return fxx


def _get_temp_day(aifs_dir, gfs_dir, target_date):
    a = build_daily_extremes(aifs_dir, start_date=str(target_date.date()), end_date=str(target_date.date()), min_timesteps=4)
    g = build_daily_extremes(gfs_dir, start_date=str(target_date.date()), end_date=str(target_date.date()), min_timesteps=4)
    common = sorted(set(a) & set(g))
    if not common:
        return None
    day = common[0]
    return day, a[day], g[day]


def _get_precip(aifs_dir, gfs_dir):
    a = load_precip_forecast_records(aifs_dir, "AIFS")
    g = load_precip_forecast_records(gfs_dir, "GFS")
    adf = pd.DataFrame({"valid_time":[r.valid_time for r in a],"location":[r.location for r in a],"value_aifs":[r.value for r in a]})
    gdf = pd.DataFrame({"valid_time":[r.valid_time for r in g],"location":[r.location for r in g],"value_gfs":[r.value for r in g]})
    adf.valid_time = pd.to_datetime(adf.valid_time).dt.tz_localize(None)
    gdf.valid_time = pd.to_datetime(gdf.valid_time).dt.tz_localize(None)
    df = adf.merge(gdf, on=["valid_time","location"], how="inner")
    if df.empty:
        return None
    # Require one common valid time for all operational locations.
    for vt in sorted(df.valid_time.unique(), reverse=True):
        sub = df[df.valid_time == vt]
        if set(sub.location) == set(LOCATIONS):
            a = {r.location: float(r.value_aifs) for r in sub.itertuples()}
            g = {r.location: float(r.value_gfs) for r in sub.itertuples()}
            return pd.Timestamp(vt), a, g
    return None


def _load_live(variable, aifs_fetcher, gfs_fetcher, aifs_dir, gfs_dir, lookback_hours, publication_lag_hours):
    aifs_mod = _load_module(Path(aifs_fetcher), "climafuse_aifs_live")
    gfs_mod = _load_module(Path(gfs_fetcher), "climafuse_gfs_live")
    now = datetime.now(timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
    allowed_hours = (0,) if variable == "temperature" else CYCLES
    candidates = _candidate_inits(now.replace(tzinfo=timezone.utc), publication_lag_hours, lookback_hours, allowed_hours)

    aifs_dir.mkdir(parents=True, exist_ok=True)
    gfs_dir.mkdir(parents=True, exist_ok=True)

    if variable == "temperature":
        for init in candidates:
            target_date = (init + timedelta(days=1)).date()
            print(f"Trying temperature cycle {init:%Y-%m-%d %H:%M} UTC -> target day {target_date}", file=sys.stderr)
            try:
                _fetch_one_init(variable, init, aifs_mod, gfs_mod, aifs_dir, gfs_dir)
                result = _get_temp_day(aifs_dir, gfs_dir, pd.Timestamp(target_date))
                if result is not None:
                    return result
            except Exception as exc:
                print(f"WARNING: cycle {init:%Y-%m-%d %H:%M} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
        raise RuntimeError("Could not obtain a common AIFS/GFS temperature forecast for the next UTC day.")

    for init in candidates:
        try:
            _fetch_one_init(variable, init, aifs_mod, gfs_mod, aifs_dir, gfs_dir)
            result = _get_precip(aifs_dir, gfs_dir)
            if result is not None:
                return result
        except Exception as exc:
            print(f"WARNING: precipitation cycle {init:%Y-%m-%d %H:%M} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise RuntimeError("Could not obtain a common AIFS/GFS precipitation forecast.")


def forecast_with_metadata(*, variable: str, model_file: str | Path = DEFAULT_MODEL,
                           aifs_dir: str | Path = DEFAULT_AIFS_CACHE,
                           gfs_dir: str | Path = DEFAULT_GFS_CACHE,
                           aifs_fetcher: str | Path = DEFAULT_AIFS_FETCHER,
                           gfs_fetcher: str | Path = DEFAULT_GFS_FETCHER,
                           bins: int = 80, live: bool = True,
                           lookback_hours: int = 30,
                           publication_lag_hours: float = 3.0):
    if variable not in VARIABLES:
        raise ValueError(f"Supported public variables: {VARIABLES}")
    model_path = Path(model_file).expanduser().resolve()
    aifs_path = Path(aifs_dir).expanduser().resolve()
    gfs_path = Path(gfs_dir).expanduser().resolve()
    models = load_models(model_path)

    if variable == "temperature":
        if "tmax" not in models or "tmin" not in models:
            raise ValueError("Retrained model artifact must contain tmax and tmin models.")
        if live:
            day, a_day, g_day = _load_live(variable, aifs_fetcher, gfs_fetcher, aifs_path, gfs_path, lookback_hours, publication_lag_hours)
        else:
            result = _get_temp_day(aifs_path, gfs_path, pd.Timestamp.max.normalize()) if False else None
            # Offline mode selects the newest common daily target day from cache.
            a = build_daily_extremes(aifs_path, start_date="2000-01-01", end_date="2100-01-01", min_timesteps=4)
            g = build_daily_extremes(gfs_path, start_date="2000-01-01", end_date="2100-01-01", min_timesteps=4)
            common = sorted(set(a) & set(g))
            if not common: raise RuntimeError("No common daily temperature forecast in cache.")
            key = common[-1]; day, a_day, g_day = key, a[key], g[key]

        output = {}
        for loc in LOCATIONS:
            output[loc] = fuse_temperature_distribution(
                location=loc,
                aifs_tmax=a_day[loc].tmax,
                gfs_tmax=g_day[loc].tmax,
                aifs_tmin=a_day[loc].tmin,
                gfs_tmin=g_day[loc].tmin,
                model_file=model_path,
                bins=bins,
            )
        return {"forecast_hour": FORECAST_HOUR, "target_date": str(day.date()), "variable": variable,
                "temperature_components": ["tmax", "tmin"], "source": {"aifs":"AI model from third-party provider", "gfs":"NWP model from third-party provider"}, "locations": output}

    if live:
        vt, av, gv = _load_live(variable, aifs_fetcher, gfs_fetcher, aifs_path, gfs_path, lookback_hours, publication_lag_hours)
    else:
        res = _get_precip(aifs_path, gfs_path)
        if res is None: raise RuntimeError("No common precipitation forecast in cache.")
        vt, av, gv = res
    output = {loc: fuse_component_distribution(variable="precipitation", location=loc, aifs_value=av[loc], gfs_value=gv[loc], model_file=model_path, bins=bins) for loc in LOCATIONS}
    return {"forecast_hour":FORECAST_HOUR,"valid_time":vt.isoformat(),"variable":variable,
            "source":{"aifs":"AI model from third-party provider","gfs":"NWP model from third-party provider"},"locations":output}


def forecast(*, variable: str, model_file: str | Path = DEFAULT_MODEL,
             aifs_dir: str | Path = DEFAULT_AIFS_CACHE,
             gfs_dir: str | Path = DEFAULT_GFS_CACHE, bins: int = 80):
    return forecast_with_metadata(variable=variable, model_file=model_file, aifs_dir=aifs_dir, gfs_dir=gfs_dir, bins=bins, live=True)["locations"]


def main():
    p=argparse.ArgumentParser(description="Real-time ClimaFuse AIFS + GFS + BMA orchestrator")
    p.add_argument("--variable",choices=VARIABLES,required=True)
    p.add_argument("--model-file",default=DEFAULT_MODEL)
    p.add_argument("--aifs-dir",default=DEFAULT_AIFS_CACHE)
    p.add_argument("--gfs-dir",default=DEFAULT_GFS_CACHE)
    p.add_argument("--aifs-fetcher",default=DEFAULT_AIFS_FETCHER)
    p.add_argument("--gfs-fetcher",default=DEFAULT_GFS_FETCHER)
    p.add_argument("--bins",type=int,default=80)
    p.add_argument("--lookback-hours",type=int,default=30)
    p.add_argument("--publication-lag-hours",type=float,default=3.0)
    p.add_argument("--offline",action="store_true")
    p.add_argument("--metadata",action="store_true")
    a=p.parse_args()
    try:
        payload=forecast_with_metadata(variable=a.variable,model_file=a.model_file,aifs_dir=a.aifs_dir,gfs_dir=a.gfs_dir,aifs_fetcher=a.aifs_fetcher,gfs_fetcher=a.gfs_fetcher,bins=a.bins,live=not a.offline,lookback_hours=a.lookback_hours,publication_lag_hours=a.publication_lag_hours)
        if not a.metadata: payload=payload["locations"]
        print(json.dumps(payload,separators=(",",":"))); return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}",file=sys.stderr); return 1

if __name__=="__main__": raise SystemExit(main())
