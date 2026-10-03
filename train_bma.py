#!/usr/bin/env python3
"""Train ClimaFuse BMA models from 2025 historical data.

Internal models:
    * tmax: model-derived daily maximum temperature vs IMD daily tmax
    * tmin: model-derived daily minimum temperature vs IMD daily tmin
    * precipitation: F24 6-hour precipitation vs IMD daily rainfall (prototype)

For temperature, the forecast value is NOT an instantaneous F24 temperature.
Instead, actual temperature forecast timesteps inside each target UTC day are
used to construct a daily maximum and minimum from a single forecast run.
The frontend later receives one temperature distribution obtained by averaging
the separately predicted Tmax/Tmin distributions.

Training default:
    2025-02-25 through 2025-09-30
Evaluation default:
    2025-10-01 through 2025-12-31
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

from bma_core import FORECAST_HOUR, LOCATIONS, MODEL_VARIABLES, fit_bma, save_models
from temperature_targets import build_daily_extremes, records_to_frame


# Preserve the old public function name for callers that need precipitation.
def _coord_name(ds: xr.Dataset, candidates):
    for name in candidates:
        if name in ds.coords:
            return name
    raise ValueError(f"Missing coordinate among {tuple(candidates)}")


def _data_var(ds: xr.Dataset, preferred: str) -> str:
    if preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    aliases = {
        "precipitation": ("tp", "apcp", "precipitation"),
        "temperature": ("temperature", "t2m"),
    }
    for name in aliases.get(preferred, ()):
        if name in ds.data_vars:
            return name
    raise ValueError(f"Could not identify {preferred}; data variables={list(ds.data_vars)}")


def _normalize_lon(lon: float, ds: xr.Dataset, lon_name: str) -> float:
    vals = np.asarray(ds[lon_name].values, dtype=float)
    if vals.size and np.nanmin(vals) >= 0 and np.nanmax(vals) > 180 and lon < 0:
        return lon % 360.0
    return lon


def load_precip_forecast_records(root: Path, model_name: str):
    files = sorted(root.rglob("*_precipitation.nc")) if root.exists() else []
    if not files:
        raise FileNotFoundError(f"No '*_precipitation.nc' files under {root}")
    rows = []
    for path in files:
        try:
            with xr.open_dataset(path) as ds:
                if "forecast_hour" not in ds.coords and "forecast_hour" not in ds.dims:
                    raise ValueError("Missing forecast_hour")
                try:
                    selected = ds.sel(forecast_hour=FORECAST_HOUR)
                except Exception as exc:
                    raise ValueError("No F24") from exc
                if "valid_time" in selected.coords:
                    vt = pd.Timestamp(np.asarray(selected["valid_time"].values).reshape(-1)[0]).tz_localize(None)
                elif "init_time" in selected.coords:
                    vt = pd.Timestamp(np.asarray(selected["init_time"].values).reshape(-1)[0]) + pd.Timedelta(hours=24)
                else:
                    raise ValueError("Missing valid_time/init_time")
                name = _data_var(selected, "precipitation")
                da = selected[name]
                lat_name = _coord_name(selected, ("latitude", "lat"))
                lon_name = _coord_name(selected, ("longitude", "lon"))
                for location, (lat, lon) in LOCATIONS.items():
                    v = float(da.sel({lat_name: lat, lon_name: _normalize_lon(lon, selected, lon_name)}, method="nearest").load().values)
                    if np.isfinite(v) and v >= 0:
                        rows.append({"target_date": vt.normalize(), "location": location, "precipitation": v})
        except Exception as exc:
            print(f"WARNING: skipping {model_name} precipitation file {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if not rows:
        raise RuntimeError(f"No usable {model_name} precipitation records found")
    return pd.DataFrame(rows)


def _load_imd_dataset(path: Path, expected_var: str) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(path)
    ds = xr.open_dataset(path)
    if expected_var not in ds.data_vars:
        if len(ds.data_vars) == 1:
            ds = ds.rename({next(iter(ds.data_vars)): expected_var})
        else:
            ds.close()
            raise ValueError(f"Expected {expected_var} in {path}; found {list(ds.data_vars)}")
    return ds


def _select_imd_spatial_cell(da: xr.DataArray, location: str, dates: list[pd.Timestamp], *, min_coverage: float = 0.50):
    lat_name = _coord_name(da.to_dataset(name=da.name or "obs"), ("lat", "latitude"))
    lon_name = _coord_name(da.to_dataset(name=da.name or "obs"), ("lon", "longitude"))
    target_lat, target_lon = LOCATIONS[location]
    target_lon = _normalize_lon(float(target_lon), da.to_dataset(name="obs"), lon_name)
    available = pd.DatetimeIndex(pd.to_datetime(da["time"].values)).normalize()
    requested = sorted({pd.Timestamp(d).normalize() for d in dates})
    mask = np.array([t in set(requested) for t in available], dtype=bool)
    if not mask.any():
        raise RuntimeError(f"No IMD dates for {location}/{da.name}")
    sampled = da.isel(time=np.flatnonzero(mask)).load()
    valid = np.isfinite(sampled) & (sampled > -900.0)
    if da.name == "rain":
        valid &= sampled >= 0.0
    coverage = valid.mean(dim="time")
    lat_values = np.asarray(da[lat_name].values, dtype=float)
    lon_values = np.asarray(da[lon_name].values, dtype=float)
    lat2, lon2 = np.meshgrid(lat_values, lon_values, indexing="ij")
    distance2 = (lat2 - float(target_lat)) ** 2 + (lon2 - float(target_lon)) ** 2
    cov_values = np.asarray(coverage.values, dtype=float)
    eligible = np.isfinite(cov_values) & (cov_values >= min_coverage)
    if eligible.any():
        score = np.where(eligible, distance2, np.inf)
    else:
        best_cov = np.nanmax(np.where(np.isfinite(cov_values), cov_values, np.nan))
        score = np.where(np.isfinite(cov_values) & np.isclose(cov_values, best_cov, rtol=0, atol=1e-12), distance2, np.inf)
    idx = np.unravel_index(np.argmin(score), score.shape)
    return lat_name, lon_name, float(lat_values[idx[0]]), float(lon_values[idx[1]]), float(cov_values[idx])


def load_imd_series(imd_dir: Path, variable: str, dates: list[pd.Timestamp]):
    if variable == "precipitation":
        path = imd_dir / "rain" / "imd_rain_2025_2025.nc"
        if not path.exists():
            matches = sorted((imd_dir / "rain").glob("*.nc"))
            if not matches: raise FileNotFoundError("No IMD rain NetCDF")
            path = matches[0]
        ds = _load_imd_dataset(path, "rain")
        try:
            da = ds["rain"]; out = {}
            for loc in LOCATIONS:
                la,lo,slat,slon,cov=_select_imd_spatial_cell(da,loc,dates)
                print(f"IMD {loc}/rain: selected grid ({slat:.2f}, {slon:.2f}) with {cov:.1%} valid coverage.", file=sys.stderr)
                s=da.sel({la:slat,lo:slon},method="nearest").load()
                times=pd.DatetimeIndex(pd.to_datetime(s.time.values)).normalize()
                for d in sorted({pd.Timestamp(x).normalize() for x in dates}):
                    idx=np.flatnonzero(times==d)
                    if len(idx)==1:
                        v=float(np.asarray(s.isel(time=int(idx[0])).values))
                        if np.isfinite(v) and v>=0: out[(loc,d)]=v
            return out
        finally:
            ds.close()

    paths={}
    for var in ("tmin","tmax"):
        path=imd_dir/var/f"imd_{var}_2025_2025.nc"
        if not path.exists():
            matches=sorted((imd_dir/var).glob("*.nc"))
            if not matches: raise FileNotFoundError(f"No IMD {var} NetCDF")
            path=matches[0]
        paths[var]=path
    dsets={v:_load_imd_dataset(paths[v],v) for v in paths}
    try:
        output={}
        for v,ds in dsets.items():
            da=ds[v]
            for loc in LOCATIONS:
                la,lo,slat,slon,cov=_select_imd_spatial_cell(da,loc,dates)
                print(f"IMD {loc}/{v}: selected grid ({slat:.2f}, {slon:.2f}) with {cov:.1%} valid coverage.", file=sys.stderr)
                s=da.sel({la:slat,lo:slon},method="nearest").load()
                times=pd.DatetimeIndex(pd.to_datetime(s.time.values)).normalize()
                for d in sorted({pd.Timestamp(x).normalize() for x in dates}):
                    idx=np.flatnonzero(times==d)
                    if len(idx)==1:
                        val=float(np.asarray(s.isel(time=int(idx[0])).values))
                        if np.isfinite(val) and val>-900: output[(loc,d,v)]=val
        return output
    finally:
        for ds in dsets.values(): ds.close()


def train_temperature(aifs_dir: Path, gfs_dir: Path, imd_dir: Path, start: str, end: str):
    af=build_daily_extremes(aifs_dir,start_date=start,end_date=end,min_timesteps=4)
    gf=build_daily_extremes(gfs_dir,start_date=start,end_date=end,min_timesteps=4)
    adf=records_to_frame(af).rename(columns={"tmax":"aifs_tmax","tmin":"aifs_tmin"})
    gdf=records_to_frame(gf).rename(columns={"tmax":"gfs_tmax","tmin":"gfs_tmin"})
    paired=adf.merge(gdf,on=["target_date","location"],how="inner",suffixes=("","_gfs_meta"))
    if paired.empty: raise RuntimeError("No overlapping AIFS/GFS daily temperature samples")
    dates=paired.target_date.tolist()
    obs=load_imd_series(imd_dir,"temperature",dates)
    paired["obs_tmax"]=[obs.get((r.location,r.target_date,"tmax"),np.nan) for r in paired.itertuples()]
    paired["obs_tmin"]=[obs.get((r.location,r.target_date,"tmin"),np.nan) for r in paired.itertuples()]
    paired=paired.dropna(subset=["aifs_tmax","gfs_tmax","obs_tmax","aifs_tmin","gfs_tmin","obs_tmin"])
    models={}; report={}
    for component, af_col,gf_col,obs_col in (("tmax","aifs_tmax","gfs_tmax","obs_tmax"),("tmin","aifs_tmin","gfs_tmin","obs_tmin")):
        models[component]={}; report[component]={}
        for location in LOCATIONS:
            rows=paired[paired.location==location]
            if len(rows)<30: raise RuntimeError(f"Only {len(rows)} usable samples for {location}/{component}")
            m=fit_bma(rows[af_col].to_numpy(),rows[gf_col].to_numpy(),rows[obs_col].to_numpy(),variable=component,location=location,forecast_hour=FORECAST_HOUR,training_start=start,training_end=end)
            models[component][location]=m
            report[component][location]={"samples":m.n_samples,"weight_aifs":m.weight_aifs,"weight_gfs":m.weight_gfs,"aifs_sigma":m.aifs_sigma,"gfs_sigma":m.gfs_sigma}
    return models,report


def train_precipitation(aifs_dir,gfs_dir,imd_dir,start,end):
    a=load_precip_forecast_records(aifs_dir,"AIFS"); g=load_precip_forecast_records(gfs_dir,"GFS")
    a["target_date"]=pd.to_datetime(a.target_date); g["target_date"]=pd.to_datetime(g.target_date)
    pair=a.merge(g,on=["target_date","location"],how="inner",suffixes=("_aifs","_gfs"))
    start_ts=pd.Timestamp(start); end_ts=pd.Timestamp(end)
    pair=pair[(pair.target_date>=start_ts)&(pair.target_date<=end_ts)].copy()
    if pair.empty: raise RuntimeError("No overlapping precipitation samples")
    obs=load_imd_series(imd_dir,"precipitation",pair.target_date.tolist())
    pair["observed"]=[obs.get((r.location,r.target_date.normalize()),np.nan) for r in pair.itertuples()]
    pair=pair.dropna(subset=["precipitation_aifs","precipitation_gfs","observed"])
    models={}; report={}
    for location in LOCATIONS:
        rows=pair[pair.location==location]
        if len(rows)<30: raise RuntimeError(f"Only {len(rows)} usable samples for {location}/precipitation")
        m=fit_bma(rows.precipitation_aifs.to_numpy(),rows.precipitation_gfs.to_numpy(),rows.observed.to_numpy(),variable="precipitation",location=location,forecast_hour=FORECAST_HOUR,training_start=start,training_end=end)
        models[location]=m; report[location]={"samples":m.n_samples,"weight_aifs":m.weight_aifs,"weight_gfs":m.weight_gfs,"aifs_sigma":m.aifs_sigma,"gfs_sigma":m.gfs_sigma}
    return models,report


def build_parser():
    p=argparse.ArgumentParser(description="Train ClimaFuse BMA: internal Tmax/Tmin + precipitation")
    p.add_argument("--aifs-dir",default="./data/regridded/aifs_2025")
    p.add_argument("--gfs-dir",default="./data/regridded/gfs_2025")
    p.add_argument("--imd-dir",default="./data/imd_2025")
    p.add_argument("--variable",choices=("temperature","precipitation","all"),default="all")
    p.add_argument("--start",default="2025-02-25")
    p.add_argument("--end",default="2025-09-30")
    p.add_argument("--output",default="./models/bma_models_2025.json")
    return p


def main():
    args=build_parser().parse_args()
    if pd.Timestamp(args.start)>pd.Timestamp(args.end):
        print("ERROR: --start must be <= --end",file=sys.stderr); return 2
    variables=("temperature","precipitation") if args.variable=="all" else (args.variable,)
    all_models={}; reports={}
    try:
        for var in variables:
            print(f"Training {var}: {args.start} -> {args.end}",file=sys.stderr)
            if var=="temperature":
                m,r=train_temperature(Path(args.aifs_dir).resolve(),Path(args.gfs_dir).resolve(),Path(args.imd_dir).resolve(),args.start,args.end)
                all_models.update(m); reports[var]=r
            else:
                m,r=train_precipitation(Path(args.aifs_dir).resolve(),Path(args.gfs_dir).resolve(),Path(args.imd_dir).resolve(),args.start,args.end)
                all_models[var]=m; reports[var]=r
        save_models(all_models,args.output,metadata={
            "training_period":[args.start,args.end],
            "forecast_hour_anchor":24,
            "model_variables":list(MODEL_VARIABLES),
            "public_temperature_output":"average of separately predicted Tmax and Tmin distributions",
            "temperature_window":"actual available model forecast timesteps within target UTC day; no interpolation",
            "temperature_min_timesteps":4,
            "temperature_models":"tmax and tmin trained independently",
            "precipitation":"F24 6-hour accumulation paired with IMD daily rain (existing prototype semantics)",
            "report":reports,
        })
        print(f"WROTE MODEL ARTIFACT: {Path(args.output).resolve()}",file=sys.stderr); return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}",file=sys.stderr); return 1

if __name__=="__main__": raise SystemExit(main())
