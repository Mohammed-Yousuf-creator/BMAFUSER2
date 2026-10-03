# ClimaFuse BMA Training — 2025

## Revised temperature design

Temperature is trained as two separate targets:

- `tmax` — daily maximum temperature
- `tmin` — daily minimum temperature

AIFS and GFS raw 2-m temperature forecasts are reduced to daily extrema using the **actual forecast timesteps present in each NetCDF run**. No hourly values are invented or interpolated.

The final public temperature forecast is produced after inference by averaging the separately predicted Tmax and Tmin distributions:

`Temperature = (Tmax + Tmin) / 2`

The frontend therefore continues to receive one public `temperature` distribution, not separate Tmax/Tmin outputs.

## Data alignment

For each target date, the trainer uses the IMD daily temperature observation window represented as 03:00 to 03:00 UTC. Because AIFS is available on 6-hour increments, the actual model points used are 06, 12, 18 and 00 UTC. The preferred source run is the 00 UTC initialization on the previous date, giving F006, F012, F018 and F024. No hourly values are invented or interpolated.

A missing AIFS or GFS case is skipped through an inner join; it is not replaced with a fabricated value.

## Training period

Default training:

`2025-02-25` through `2025-09-30`

Recommended held-out evaluation:

`2025-10-01` through `2025-12-31`

## Train command

From the backend/script directory:

```powershell
python train_bma.py `
  --aifs-dir ./data/regridded/aifs_2025 `
  --gfs-dir ./data/regridded/gfs_2025 `
  --imd-dir ./data/imd_2025 `
  --variable all `
  --start 2025-02-25 `
  --end 2025-09-30 `
  --output ./models/bma_models_2025.json
```

If PowerShell line continuation is inconvenient, use one line:

```powershell
python train_bma.py --aifs-dir ./data/regridded/aifs_2025 --gfs-dir ./data/regridded/gfs_2025 --imd-dir ./data/imd_2025 --variable all --start 2025-02-25 --end 2025-09-30 --output ./models/bma_models_2025.json
```

## Required temperature source data

AIFS and GFS temperature files must contain enough actual forecast timesteps to form daily extrema. The fetchers can be asked for the needed 6-hour leads; the AIFS fetcher accepts forecast hours only in 6-hour increments.

For a full historical refetch, a practical example is:

```powershell
python ecmwf_aifs_fetcher.py --start 2025-02-24 --end 2025-09-29 --variable temperature --cycles 0 --fxx 6 12 18 24 --save-dir ./data/aifs
```

```powershell
python noaa_gfs_fetcher.py --start 2025-02-24 --end 2025-09-29 --variable temperature --cycles 0 --fxx 6 12 18 24 --save-dir ./data/gfs
```

The training pipeline only uses the timesteps required for a target day and does not create missing timesteps.

## BMA fitting

For each location, the trainer fits:

- one BMA model for Tmax;
- one BMA model for Tmin;
- one BMA model for precipitation.

Each BMA model estimates an AIFS weight, GFS weight, calibration intercept/slope and uncertainty for its component. The two temperature models are independent during fitting.

## Evaluation

Use the held-out period:

```powershell
python evaluate_bma.py --model-file ./models/bma_models_2025.json --aifs-dir ./data/regridded/aifs_2025 --gfs-dir ./data/regridded/gfs_2025 --imd-dir ./data/imd_2025 --variable all --start 2025-10-01 --end 2025-12-31 --csv ./reports/bma_evaluation.csv --json ./reports/bma_evaluation.json
```

For temperature, evaluation uses the public daily-temperature output `(BMA Tmax + BMA Tmin)/2` against `(IMD Tmax + IMD Tmin)/2` and separately checks the 90% interval of that public distribution.

Do not report an evaluation period that overlaps the saved model's training dates as a held-out test result.
