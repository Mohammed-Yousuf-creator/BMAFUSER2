#!/usr/bin/env python3
"""Download NOAA GFS 0.25-degree fields with Herbie and save NetCDF.

Variables
---------
temperature    2-m temperature, converted to degC
precipitation  6-hour APCP accumulation, converted to mm

The fetcher uses Herbie's GFS ``pgrb2.0p25`` product. GFS is available 4 times
per day (00/06/12/18 UTC); the script accepts an inclusive initialization range.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_CYCLES = (0, 12)
DEFAULT_FXX = (6, 12, 24, 48, 72)
DEFAULT_PRIORITY = ("aws", "google", "nomads", "azure")
GFS_PRODUCT = "pgrb2.0p25"


def require_deps():
    try:
        from herbie import Herbie
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("Install the dependencies from requirements.txt first.") from exc
    return Herbie, xr


def parse_datetime(value: str) -> datetime:
    value = value.strip().replace("Z", "")
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date/time '{value}'.") from exc
    if dt.tzinfo is not None:
        raise argparse.ArgumentTypeError("Supply GFS archive times in UTC without a timezone offset.")
    return dt.replace(second=0, microsecond=0)


def iter_initializations(start: datetime, end: datetime, cycles: Sequence[int]):
    if start > end:
        raise ValueError("--start must be earlier than or equal to --end.")
    cycles = sorted(set(cycles))
    invalid = [c for c in cycles if c not in (0, 6, 12, 18)]
    if invalid:
        raise ValueError(f"Cycles must be one of 00, 06, 12, 18 UTC. Invalid: {invalid}")
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    last = end.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= last:
        for cycle in cycles:
            init = day + timedelta(hours=cycle)
            if start <= init <= end:
                yield init
        day += timedelta(days=1)


def _pick_data_var(ds, preferred):
    for name in preferred:
        if name in ds.data_vars:
            return name
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise ValueError(f"Could not identify expected variable. Found: {list(ds.data_vars)}")


def _gfs_apcp_search(fxx: int) -> str:
    if fxx <= 0:
        raise ValueError("GFS precipitation requires positive forecast hours.")
    # GFS APCP is published as accumulation intervals. For the selected
    # 6-hour leads this selects 0-6, 6-12, 18-24, 42-48, etc.
    start = 6 * ((fxx - 1) // 6)
    return fr":APCP:surface:{start}-{fxx} hour acc fcst:"


def _load_gfs(Herbie, xr, init, fxx, variable, priority, save_dir):
    H = Herbie(
        init,
        model="gfs",
        product=GFS_PRODUCT,
        fxx=fxx,
        priority=list(priority),
        save_dir=save_dir,
        verbose=True,
    )
    if H.grib is None:
        raise FileNotFoundError(f"GFS file unavailable for {init:%Y-%m-%d %H:%M} UTC F{fxx:03d}")

    if variable == "temperature":
        ds = H.xarray(":TMP:2 m above")
        name = _pick_data_var(ds, ("t2m",))
        da = ds[name].astype("float32") - np.float32(273.15)
        da.name = "temperature"
        da.attrs.update({"long_name": "2-m air temperature", "units": "degC"})

    elif variable == "precipitation":
        search = _gfs_apcp_search(fxx)
        ds = H.xarray(search)
        name = _pick_data_var(ds, ("tp", "apcp", "prate"))
        da = ds[name].astype("float32")
        # GFS APCP is kg m-2 for precipitation water equivalent, numerically mm.
        da.name = "precipitation"
        da.attrs.update({
            "long_name": "6-hour GFS accumulated precipitation",
            "units": "mm",
            "accumulation_period_hours": 6,
            "herbie_search": search,
        })

    else:
        raise ValueError(variable)

    if "valid_time" in da.coords:
        da = da.drop_vars("valid_time")
    return da.expand_dims(forecast_hour=[fxx])


def fetch_gfs(
    start: datetime,
    end: datetime,
    variable: str,
    *,
    cycles: Sequence[int],
    fxx: Sequence[int],
    priority: Sequence[str],
    save_dir: str | Path,
    overwrite: bool = False,
    fail_fast: bool = False,
):
    Herbie, xr = require_deps()
    out_root = Path(save_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    fxx = sorted(set(fxx))
    if not fxx or any(h < 0 or h > 384 for h in fxx):
        raise ValueError("GFS forecast hours must be between 0 and 384.")

    inits = list(iter_initializations(start, end, cycles))
    print(f"GFS product: {GFS_PRODUCT}")
    print(f"Variable: {variable}")
    print(f"Initialization runs: {len(inits)}")
    print(f"Forecast hours: {fxx}")

    for init in inits:
        pieces = {}
        failed = False
        for hour in fxx:
            try:
                print(f"Fetching GFS {init:%Y-%m-%d %Hz} F{hour:03d} -> {variable}")
                pieces[hour] = _load_gfs(Herbie, xr, init, hour, variable, priority, out_root)
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
                failed = True
                if fail_fast:
                    raise

        if not pieces:
            continue

        data = xr.concat([pieces[h] for h in fxx if h in pieces], dim="forecast_hour")
        data = data.sortby("forecast_hour").astype("float32")
        data = data.assign_coords(valid_time=("forecast_hour", np.array([
            np.datetime64(init + timedelta(hours=int(h)), "ns")
            for h in data.forecast_hour.values
        ])))
        data = data.assign_coords(init_time=np.datetime64(init, "ns"))

        ds_out = data.to_dataset(name=data.name or variable)
        ds_out.attrs.update({
            "model": "NOAA GFS",
            "product": GFS_PRODUCT,
            "source_library": "Herbie",
            "initialization_time_utc": init.isoformat(sep=" "),
            "requested_variable": variable,
            "forecast_hours": ",".join(str(int(x)) for x in data.forecast_hour.values),
        })

        cycle_dir = out_root / f"{init:%Y%m%d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        output = cycle_dir / f"gfs_{GFS_PRODUCT}_{init:%Y%m%d%H}_{variable}.nc"
        if output.exists() and not overwrite:
            print(f"SKIP (exists): {output}")
        else:
            variable_name = list(ds_out.data_vars)[0]
            encoding = {variable_name: {"dtype": "float32", "zlib": True, "complevel": 4}}
            ds_out.to_netcdf(output, engine="netcdf4", encoding=encoding)
            print(f"WROTE: {output}")

        if failed:
            print(f"Completed {init:%Y-%m-%d %Hz} with some failed source requests.", file=sys.stderr)


def build_parser():
    p = argparse.ArgumentParser(description="Download NOAA GFS 0.25-degree temperature or precipitation.")
    p.add_argument("--start", required=True, type=parse_datetime)
    p.add_argument("--end", required=True, type=parse_datetime)
    p.add_argument("--variable", required=True, choices=("temperature", "precipitation"))
    p.add_argument("--cycles", type=int, nargs="+", default=list(DEFAULT_CYCLES), choices=(0, 6, 12, 18))
    p.add_argument("--fxx", type=int, nargs="+", default=list(DEFAULT_FXX))
    p.add_argument("--priority", nargs="+", default=list(DEFAULT_PRIORITY))
    p.add_argument("--save-dir", default="./data/gfs")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        fetch_gfs(
            args.start,
            args.end,
            args.variable,
            cycles=args.cycles,
            fxx=args.fxx,
            priority=args.priority,
            save_dir=args.save_dir,
            overwrite=args.overwrite,
            fail_fast=args.fail_fast,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
