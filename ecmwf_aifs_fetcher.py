#!/usr/bin/env python3
"""Download ECMWF AIFS fields with Herbie and save standardized NetCDF files.

Variables
---------
temperature    2-m temperature, converted to degC
precipitation  6-hour precipitation accumulation, converted to mm

The script writes one NetCDF file per initialization (date + cycle) so forecast
hours are kept together on a single ``forecast_hour`` dimension.

Network resilience
------------------
Requests are paced (``--pause``) and transient failures (HTTP 503 "Slow Down",
timeouts, Herbie's ``KeyError: 'href'``) are retried with exponential backoff,
rotating through the mirrors in ``--priority`` and forcing a clean re-download.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

AIFS_FXX = tuple(range(0, 361, 6))
DEFAULT_CYCLES = (0, 12)
DEFAULT_FXX = (6, 12, 24, 48, 72)
DEFAULT_PRIORITY = ("azure", "aws", "ecmwf", "google")

# Extra delay (seconds) added between requests after S3 throttles us, and
# slowly removed again as requests succeed.
_THROTTLE = {"extra": 0.0}
MAX_EXTRA_PAUSE = 30.0
MAX_BACKOFF = 300.0


def _install_http_retries(total: int = 6, backoff: float = 2.0) -> None:
    """Make every ``requests`` call (including Herbie's) retry 5xx responses.

    Herbie calls ``requests.get`` directly, and that creates a fresh Session each
    time, so we patch ``Session.__init__`` to mount a retrying adapter. A 503
    "Slow Down" is then retried on just the failing HTTP call (honouring any
    Retry-After header) instead of rebuilding the whole Herbie lookup.
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    if getattr(requests.Session, "_aifs_retry_patched", False):
        return
    original_init = requests.Session.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        retry = Retry(
            total=total,
            connect=total,
            read=total,
            status=total,
            backoff_factor=backoff,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=None,  # retry every method, including range GETs
            respect_retry_after_header=True,
            raise_on_status=False,  # hand back the final response -> HTTPError
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.mount("https://", adapter)
        self.mount("http://", adapter)

    requests.Session.__init__ = patched_init
    requests.Session._aifs_retry_patched = True


def require_deps():
    try:
        from herbie import Herbie
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies. Install requirements.txt and ESMPy as described there."
        ) from exc
    _install_http_retries()
    return Herbie, xr


def parse_datetime(value: str) -> datetime:
    value = value.strip().replace("Z", "")
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date/time '{value}'. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM."
        ) from exc
    if dt.tzinfo is not None:
        raise argparse.ArgumentTypeError("Supply AIFS archive times in UTC without a timezone offset.")
    return dt.replace(second=0, microsecond=0)


def parse_fxx(values: Sequence[int] | None) -> list[int]:
    result = list(DEFAULT_FXX if values is None else values)
    if not result:
        raise ValueError("At least one forecast hour is required.")
    invalid = [f for f in result if f not in AIFS_FXX]
    if invalid:
        raise ValueError(f"AIFS forecast hours must be multiples of 6 from 0 to 360. Invalid: {invalid}")
    return sorted(set(result))


def iter_initializations(start: datetime, end: datetime, cycles: Iterable[int]):
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


def _pick_data_var(ds, preferred: Sequence[str]) -> str:
    for name in preferred:
        if name in ds.data_vars:
            return name
    candidates = list(ds.data_vars)
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Could not identify expected data variable. Found: {candidates}")


def _aifs_precip_support_hours(requested: Sequence[int]) -> list[int]:
    """Return lead times needed to turn cumulative AIFS tp into 6-hour totals.

    F000 is never needed: the first 6-hour window is just the F006 cumulative
    value, so skipping it saves a request per run.
    """
    supports = set()
    for hour in requested:
        if hour <= 0:
            raise ValueError("Precipitation requires positive forecast hours.")
        supports.add(hour)
        if hour - 6 > 0:
            supports.add(hour - 6)
    return sorted(supports)


def _is_transient(exc: Exception) -> bool:
    """Errors worth retrying: throttling, timeouts, and Herbie's 'href' KeyError."""
    if isinstance(exc, KeyError) and exc.args and exc.args[0] == "href":
        return True
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in msg
        for marker in (
            "503",
            "429",
            "slow down",
            "timed out",
            "timeout",
            "connection",
            "throttl",
            "href",
        )
    )


def _load_aifs(
    Herbie,
    xr,
    init: datetime,
    fxx: int,
    variable: str,
    priority,
    save_dir: Path,
    overwrite: bool = False,
):
    H = Herbie(
        init,
        model="aifs",
        product="oper",
        fxx=fxx,
        priority=list(priority),
        save_dir=save_dir,
        overwrite=overwrite,
        verbose=True,
    )
    if H.grib is None:
        raise FileNotFoundError(f"AIFS file unavailable for {init:%Y-%m-%d %H:%M} UTC F{fxx:03d}")

    if variable == "temperature":
        ds = H.xarray(":2t:")
        name = _pick_data_var(ds, ("t2m",))
        da = ds[name].astype("float32") - np.float32(273.15)
        da.name = "temperature"
        da.attrs.update({"long_name": "2-m air temperature", "units": "degC"})

    elif variable == "precipitation":
        ds = H.xarray(":tp:")
        name = _pick_data_var(ds, ("tp",))
        da = ds[name].astype("float32")
        # ECMWF AIFS total precipitation is a depth in metres. Convert to mm.
        units = str(da.attrs.get("units", "m")).lower()
        if units in {"m", "meter", "metre", "meters", "metres"}:
            da = da * np.float32(1000.0)
        da.name = "precipitation_cumulative"
        da.attrs.update({"long_name": "AIFS total precipitation accumulated since initialization", "units": "mm"})

    else:
        raise ValueError(variable)

    # Herbie/xarray provides scalar valid_time. The standardized schema uses
    # valid_time as a 1-D coordinate along forecast_hour.
    if "valid_time" in da.coords:
        da = da.drop_vars("valid_time")
    da = da.expand_dims(forecast_hour=[fxx])
    return da


# Per-mirror cooldown: source name -> time.monotonic() until which it is avoided.
# A mirror that returns 503 is demoted to the back of Herbie's priority list so
# the next attempt (and the next files) use a different mirror if one has the file.
_MIRROR_COOLDOWN: dict[str, float] = {}
MIRROR_COOLDOWN_SECONDS = 240.0
_SOURCE_HOSTS = (
    ("amazonaws.com", "aws"),
    ("windows.net", "azure"),
    ("googleapis.com", "google"),
    ("ecmwf.int", "ecmwf"),
)


def _source_from_exc(exc: Exception) -> str | None:
    """Work out which mirror an error came from using the URL in its message."""
    text = str(exc).lower()
    for host, name in _SOURCE_HOSTS:
        if host in text:
            return name
    return None


def _ordered_mirrors(priority: Sequence[str]) -> list[str]:
    """Healthy mirrors first (in the user's order), cooling-down mirrors last."""
    now = time.monotonic()
    ready = [s for s in priority if _MIRROR_COOLDOWN.get(s, 0.0) <= now]
    cooling = sorted(
        (s for s in priority if s not in ready), key=lambda s: _MIRROR_COOLDOWN[s]
    )
    return ready + cooling


def _load_aifs_retry(
    Herbie,
    xr,
    init: datetime,
    fxx: int,
    variable: str,
    priority: Sequence[str],
    save_dir: Path,
    retries: int = 4,
    base_delay: float = 20.0,
    debug: bool = False,
):
    """Load one field, retrying transient errors and switching away from throttled mirrors."""
    priority = list(priority)
    force = False  # only force a fresh download after a non-throttle error
    for attempt in range(retries + 1):
        prio = _ordered_mirrors(priority)
        try:
            result = _load_aifs(
                Herbie,
                xr,
                init,
                fxx,
                variable,
                prio,
                save_dir,
                overwrite=force,
            )
            _THROTTLE["extra"] = max(0.0, _THROTTLE["extra"] - 1.0)
            return result
        except Exception as exc:  # noqa: BLE001
            if debug:
                traceback.print_exc()
            if attempt == retries or not _is_transient(exc):
                raise
            # A 503 means "slow down", not "bad file": reuse the cached index.
            force = isinstance(exc, KeyError)

            src = _source_from_exc(exc)
            if src:
                _MIRROR_COOLDOWN[src] = time.monotonic() + MIRROR_COOLDOWN_SECONDS
            now = time.monotonic()
            alternatives = [
                s for s in priority
                if s != src and _MIRROR_COOLDOWN.get(s, 0.0) <= now
            ]
            if alternatives:
                # Another mirror may have the file: switch quickly, no long wait.
                delay = 3.0 + random.uniform(0, 2)
                action = f"switching mirror (avoiding '{src}', next '{alternatives[0]}')"
            else:
                _THROTTLE["extra"] = min(MAX_EXTRA_PAUSE, _THROTTLE["extra"] + 3.0)
                delay = min(MAX_BACKOFF, base_delay * (2 ** attempt)) + random.uniform(0, 5)
                action = "all mirrors throttled/unavailable, backing off"
            print(
                f"  transient error ({type(exc).__name__}: {exc}); "
                f"retry {attempt + 1}/{retries} in {delay:.0f}s, {action}",
                file=sys.stderr,
            )
            time.sleep(delay)


def _load_with_cooldown(
    Herbie,
    xr,
    init,
    fxx,
    variable,
    priority,
    save_dir,
    *,
    retries: int,
    cooldown_min: float,
    max_cooldowns: int,
    debug: bool = False,
):
    """Retry one file; if throttling persists, cool down for a while and try again."""
    for round_ in range(max_cooldowns + 1):
        try:
            return _load_aifs_retry(
                Herbie, xr, init, fxx, variable, priority, save_dir,
                retries=retries, debug=debug,
            )
        except Exception as exc:  # noqa: BLE001
            if round_ == max_cooldowns or not _is_transient(exc):
                raise
            print(
                f"  still throttled after {retries} retries; cooling down "
                f"{cooldown_min:g} min (round {round_ + 1}/{max_cooldowns})",
                file=sys.stderr,
            )
            time.sleep(cooldown_min * 60)


def fetch_aifs(
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
    retries: int = 4,
    pause: float = 3.0,
    cooldown_min: float = 10.0,
    max_cooldowns: int = 3,
    max_consecutive_failures: int = 3,
    debug: bool = False,
):
    # xarray is intentionally imported only after CLI parsing.
    Herbie, xr = require_deps()
    out_root = Path(save_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # --priority is a preference order, not a restriction: the remaining mirrors
    # are appended as fallbacks so a throttled source can't trap the run.
    priority = list(priority) + [m for m in DEFAULT_PRIORITY if m not in priority]
    print(f"Mirror order: {priority}")

    requested = parse_fxx(fxx)
    fetch_hours = _aifs_precip_support_hours(requested) if variable == "precipitation" else requested

    inits = list(iter_initializations(start, end, cycles))
    if not inits:
        raise ValueError("No requested AIFS initialization cycles fall inside the range.")

    print(f"AIFS variable: {variable}")
    print(f"Initialization runs: {len(inits)}")
    print(f"Requested forecast hours: {requested}")
    if variable == "precipitation":
        print(f"Internal precipitation support hours: {fetch_hours}")

    consecutive_failures = 0
    failed_inits = []

    for init in inits:
        output = out_root / f"{init:%Y%m%d}" / f"aifs_oper_{init:%Y%m%d%H}_{variable}.nc"
        if output.exists() and not overwrite:
            print(f"SKIP (exists): {output}")
            continue

        pieces = {}
        failed = False
        for hour in fetch_hours:
            try:
                print(f"Fetching AIFS {init:%Y-%m-%d %Hz} F{hour:03d} -> {variable}")
                pieces[hour] = _load_with_cooldown(
                    Herbie,
                    xr,
                    init,
                    hour,
                    variable,
                    priority,
                    out_root,
                    retries=retries,
                    cooldown_min=cooldown_min,
                    max_cooldowns=max_cooldowns,
                    debug=debug,
                )
                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
                failed = True
                if _is_transient(exc):
                    consecutive_failures += 1
                if fail_fast:
                    raise
                if variable == "precipitation":
                    # A 6-hour total needs every support lead; skip the rest.
                    break
            finally:
                time.sleep(pause + _THROTTLE["extra"])  # stay under S3's rate limit

        if failed:
            failed_inits.append(init)
        if consecutive_failures >= max_consecutive_failures:
            raise RuntimeError(
                f"{consecutive_failures} files in a row failed with throttling/network "
                "errors even after cooldowns. Stop, wait 30+ minutes (or change "
                "network), then rerun the same command; finished files are skipped."
            )

        if not pieces:
            continue

        if variable == "precipitation":
            cumulative = xr.concat([pieces[h] for h in sorted(pieces)], dim="forecast_hour")
            # For the requested horizon, calculate a 6-hour accumulation ending
            # at that forecast hour. This makes AIFS precipitation semantics align
            # with the GFS APCP 6-hour accumulation used by the GFS fetcher.
            result_parts = []
            for hour in requested:
                if hour not in cumulative.forecast_hour.values:
                    continue
                previous = hour - 6
                current = cumulative.sel(forecast_hour=hour)
                if previous == 0:
                    prev = 0.0
                elif previous in cumulative.forecast_hour.values:
                    prev = cumulative.sel(forecast_hour=previous)
                else:
                    print(f"WARNING: Missing AIFS support lead F{previous:03d}; skipping F{hour:03d}.")
                    continue
                part = current - prev
                part = part.expand_dims(forecast_hour=[hour])
                part.name = "precipitation"
                part.attrs.update({
                    "long_name": "6-hour precipitation accumulation",
                    "units": "mm",
                    "accumulation_period_hours": 6,
                })
                result_parts.append(part)
            if not result_parts:
                continue
            data = xr.concat(result_parts, dim="forecast_hour")
        else:
            data = xr.concat([pieces[h] for h in requested if h in pieces], dim="forecast_hour")

        data = data.sortby("forecast_hour")
        data = data.assign_coords(valid_time=("forecast_hour", np.array([
            np.datetime64(init + timedelta(hours=int(h)), "ns")
            for h in data.forecast_hour.values
        ])))
        data = data.assign_coords(init_time=np.datetime64(init, "ns"))
        data = data.astype("float32")

        ds_out = data.to_dataset(name=data.name or variable)
        ds_out.attrs.update({
            "model": "ECMWF AIFS",
            "product": "oper",
            "source_library": "Herbie",
            "initialization_time_utc": init.isoformat(sep=" "),
            "requested_variable": variable,
            "forecast_hours": ",".join(str(int(x)) for x in data.forecast_hour.values),
        })

        output.parent.mkdir(parents=True, exist_ok=True)
        encoding = {ds_out.data_vars[list(ds_out.data_vars)[0]].name: {
            "dtype": "float32", "zlib": True, "complevel": 4,
        }}
        ds_out.to_netcdf(output, engine="netcdf4", encoding=encoding)
        print(f"WROTE: {output}")

        if failed:
            print(f"Completed {init:%Y-%m-%d %Hz} with some failed source requests.", file=sys.stderr)

    if failed_inits:
        print(
            f"\n{len(failed_inits)} initialization(s) had failures; rerun the same command "
            "to fill the gaps:",
            file=sys.stderr,
        )
        for init in failed_inits:
            print(f"  {init:%Y-%m-%d %Hz}", file=sys.stderr)


def build_parser():
    p = argparse.ArgumentParser(description="Download ECMWF AIFS temperature or precipitation.")
    p.add_argument("--start", required=True, type=parse_datetime)
    p.add_argument("--end", required=True, type=parse_datetime)
    p.add_argument("--variable", required=True, choices=("temperature", "precipitation"))
    p.add_argument("--cycles", type=int, nargs="+", default=list(DEFAULT_CYCLES), choices=(0, 6, 12, 18))
    p.add_argument("--fxx", type=int, nargs="+", default=list(DEFAULT_FXX))
    p.add_argument("--priority", nargs="+", default=list(DEFAULT_PRIORITY),
                   help="Preferred mirror order; unlisted mirrors are added as fallbacks")
    p.add_argument("--save-dir", default="./data/aifs")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--retries", type=int, default=4, help="Retries per file on transient errors")
    p.add_argument("--cooldown", type=float, default=10.0, help="Minutes to wait when throttling persists")
    p.add_argument("--max-cooldowns", type=int, default=3, help="Cooldown rounds per file before giving up on it")
    p.add_argument("--pause", type=float, default=3.0, help="Seconds to sleep between requests")
    p.add_argument("--debug", action="store_true", help="Print full tracebacks for failed requests")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        fetch_aifs(
            args.start,
            args.end,
            args.variable,
            cycles=args.cycles,
            fxx=args.fxx,
            priority=args.priority,
            save_dir=args.save_dir,
            overwrite=args.overwrite,
            fail_fast=args.fail_fast,
            retries=args.retries,
            pause=args.pause,
            cooldown_min=args.cooldown,
            max_cooldowns=args.max_cooldowns,
            debug=args.debug,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())