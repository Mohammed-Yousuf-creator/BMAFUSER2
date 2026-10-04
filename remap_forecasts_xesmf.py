#!/usr/bin/env python3
"""Regrid AIFS/GFS NetCDF forecasts onto a common target grid with xESMF.

The input NetCDF files are the outputs of ecmwf_aifs_fetcher.py and
noaa_gfs_fetcher.py. The script recursively finds *.nc files and writes the
same directory structure underneath --output-dir.

Target grid can be supplied as a NetCDF file containing 1-D latitude/longitude
coordinates, or generated from lat/lon bounds and steps.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

CONSERVATIVE_METHODS = ("conservative", "conservative_normed")


def require_deps():
    try:
        import xarray as xr
        import xesmf as xe
    except ImportError as exc:
        raise RuntimeError(
            "xarray/xESMF is not installed. Install the requirements and ESMPy as described there."
        ) from exc
    return xr, xe


def _coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords:
            return name
        if name in ds.variables:
            return name
    raise ValueError(f"Could not find any of {candidates} in {list(ds.coords)} / {list(ds.variables)}")


def standardize_grid(ds):
    lat_name = _coord_name(ds, ("lat", "latitude"))
    lon_name = _coord_name(ds, ("lon", "longitude"))
    rename = {}
    if lat_name != "lat":
        rename[lat_name] = "lat"
    if lon_name != "lon":
        rename[lon_name] = "lon"
    ds = ds.rename(rename)

    if ds["lat"].ndim != 1 or ds["lon"].ndim != 1:
        raise ValueError("This implementation expects 1-D rectilinear latitude/longitude coordinates.")

    # Normalize longitude into [-180, 180) so AIFS and GFS (0..360) are
    # represented consistently.
    lon = ((ds["lon"] + 180) % 360) - 180
    ds = ds.assign_coords(lon=lon).sortby("lon")
    ds = ds.sortby("lat")
    return ds


def make_target_grid(xr, args):
    """Return a Dataset holding only the 1-D ``lat`` / ``lon`` target coordinates."""
    if args.target_grid:
        with xr.open_dataset(args.target_grid) as target:
            target = standardize_grid(target.load())
        lat, lon = target["lat"].values, target["lon"].values
    else:
        required = [
            args.lat_min, args.lat_max, args.lat_step,
            args.lon_min, args.lon_max, args.lon_step,
        ]
        if any(v is None for v in required):
            raise ValueError(
                "Supply --target-grid OR all of --lat-min --lat-max --lat-step --lon-min --lon-max --lon-step."
            )
        lat = np.arange(args.lat_min, args.lat_max + args.lat_step * 0.5, args.lat_step, dtype=np.float64)
        lon = np.arange(args.lon_min, args.lon_max + args.lon_step * 0.5, args.lon_step, dtype=np.float64)

    grid = xr.Dataset({"lat": ("lat", np.asarray(lat, dtype=np.float64)),
                       "lon": ("lon", np.asarray(lon, dtype=np.float64))})
    # Same normalisation as the source files (lon in [-180, 180), ascending).
    return standardize_grid(grid)


def choose_method(variable_names, requested):
    if requested != "auto":
        return requested
    # Conservative-normalized remapping is preferable for precipitation depth
    # fields; bilinear is the default for temperature.
    if "precipitation" in variable_names:
        return "conservative_normed"
    return "bilinear"


def _edges(centers, clip=None):
    """Cell edges (n+1 values) from 1-D cell centres, optionally clipped."""
    c = np.asarray(centers, dtype=np.float64)
    if c.size < 2:
        raise ValueError("Need at least 2 grid points along each axis to build cell bounds.")
    mid = (c[:-1] + c[1:]) / 2.0
    edges = np.concatenate(([c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]))
    if clip is not None:
        edges = np.clip(edges, clip[0], clip[1])
    return edges


def grid_for_regrid(xr, ds, need_bounds):
    """Minimal lat/lon grid for xESMF; conservative methods also get explicit bounds.

    Latitude bounds are clipped to +-90 so global grids with points exactly on
    the poles do not produce cells that extend beyond the poles.
    """
    grid = xr.Dataset({"lat": ("lat", ds["lat"].values), "lon": ("lon", ds["lon"].values)})
    if need_bounds:
        grid["lat_b"] = ("lat_b", _edges(ds["lat"].values, clip=(-90.0, 90.0)))
        grid["lon_b"] = ("lon_b", _edges(ds["lon"].values))
    return grid


def is_global_lon(lon):
    lon = np.asarray(lon, dtype=np.float64)
    if lon.size < 2:
        return False
    step = float(np.median(np.diff(np.sort(lon))))
    return (lon.max() - lon.min() + step) >= 359.0


def weight_name(source_ds, target, method, weight_dir, periodic):
    """Weight file name that depends on the full source/target coordinates.

    Hashing only grid shapes (as before) let two different target grids with the
    same shape silently share one weights file.
    """
    h = hashlib.sha1()
    for arr in (source_ds["lat"].values, source_ds["lon"].values,
                target["lat"].values, target["lon"].values):
        a = np.ascontiguousarray(arr, dtype=np.float64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    h.update(f"{method}|periodic={bool(periodic)}".encode())
    return weight_dir / f"weights_{method}_{h.hexdigest()[:12]}.nc"


def remap_file(xr, xe, input_path, output_path, target, args, weight_dir, cache):
    with xr.open_dataset(input_path) as ds:
        ds = standardize_grid(ds)
        variable_names = list(ds.data_vars)
        method = choose_method(variable_names, args.method)
        periodic = args.periodic if args.periodic is not None else is_global_lon(ds["lon"].values)
        weight_file = weight_name(ds, target, method, weight_dir, periodic)

        # Building a Regridder is expensive; reuse it for every file on the same grid.
        regridder = cache.get(weight_file)
        if regridder is None:
            weight_dir.mkdir(parents=True, exist_ok=True)
            need_bounds = method in CONSERVATIVE_METHODS
            regridder = xe.Regridder(
                grid_for_regrid(xr, ds, need_bounds),
                grid_for_regrid(xr, target, need_bounds),
                method,
                periodic=bool(periodic),
                filename=str(weight_file),
                reuse_weights=weight_file.exists(),
                # Target points outside the source grid become NaN instead of 0
                # (0 would look like a valid 0 degC / 0 mm value).
                unmapped_to_nan=True,
            )
            cache[weight_file] = regridder

        out = regridder(ds, keep_attrs=True)

        # Keep scalar/non-spatial coordinates (init_time, valid_time, ...).
        for name, coord in ds.coords.items():
            if name not in out.coords and "lat" not in coord.dims and "lon" not in coord.dims:
                out = out.assign_coords({name: coord})

        out.attrs.update({
            "regridded_with": "xESMF",
            "regridding_method": method,
            "periodic_longitude": int(bool(periodic)),
            "target_grid_lat_points": int(target.sizes["lat"]),
            "target_grid_lon_points": int(target.sizes["lon"]),
        })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        enc = {
            name: {"dtype": "float32", "zlib": True, "complevel": 4}
            for name in out.data_vars
        }
        out.astype("float32").to_netcdf(output_path, engine="netcdf4", encoding=enc)

    return method, weight_file


def build_parser():
    p = argparse.ArgumentParser(description="Regrid AIFS/GFS NetCDF forecasts with xESMF.")
    p.add_argument("--input-dir", required=True, help="Root containing AIFS/GFS NetCDF outputs.")
    p.add_argument("--output-dir", required=True, help="Root for regridded NetCDF outputs.")
    p.add_argument("--target-grid", help="NetCDF containing target 1-D lat/lon coordinates.")
    p.add_argument("--lat-min", type=float)
    p.add_argument("--lat-max", type=float)
    p.add_argument("--lat-step", type=float)
    p.add_argument("--lon-min", type=float)
    p.add_argument("--lon-max", type=float)
    p.add_argument("--lon-step", type=float)
    p.add_argument(
        "--method",
        choices=("auto", "bilinear", "conservative", "conservative_normed", "patch", "nearest_s2d"),
        default="auto",
    )
    p.add_argument(
        "--periodic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Treat longitude as cyclic. Default: auto-detect (on when the source grid is global).",
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        xr, xe = require_deps()
        input_dir = Path(args.input_dir).expanduser().resolve()
        output_dir = Path(args.output_dir).expanduser().resolve()
        if not input_dir.exists():
            raise FileNotFoundError(input_dir)
        target = make_target_grid(xr, args)
        files = sorted(input_dir.rglob("*.nc"))
        # Don't re-process previous outputs (and weight files) if --output-dir
        # lives inside --input-dir.
        files = [f for f in files if output_dir not in f.parents]
        if not files:
            raise FileNotFoundError(f"No NetCDF files found under {input_dir}")

        weight_dir = output_dir / "weights"
        cache: dict = {}
        count = 0
        failed = []
        for path in files:
            rel = path.relative_to(input_dir)
            out = output_dir / rel
            if out.exists() and not args.overwrite:
                print(f"SKIP (exists): {out}")
                continue
            print(f"REGRID: {path} -> {out}")
            try:
                method, weights = remap_file(xr, xe, path, out, target, args, weight_dir, cache)
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
                failed.append(path)
                out.unlink(missing_ok=True)  # never leave a half-written output
                continue
            print(f"  method={method}; weights={weights}")
            count += 1

        print(f"Completed {count} files.")
        if failed:
            print(f"{len(failed)} file(s) failed:", file=sys.stderr)
            for f in failed:
                print(f"  {f}", file=sys.stderr)
            return 1
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())