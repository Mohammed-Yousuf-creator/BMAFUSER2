#!/usr/bin/env python3
"""Evaluate the revised ClimaFuse BMA model per location.

Temperature is evaluated as the public daily temperature output:
    0.5 * (BMA Tmax + BMA Tmin)
The underlying Tmax and Tmin BMA models are still evaluated separately, then
combined for the final temperature point estimate and 90% interval.

Default held-out period:
    2025-10-01 -> 2025-12-31

Use only after retraining with the default training window
2025-02-25 -> 2025-09-30.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

from bma_core import LOCATIONS, VARIABLES, load_models
from train_bma import load_precip_forecast_records, load_imd_series
from temperature_targets import build_daily_extremes, records_to_frame
from bma_core import _working_transform, _inverse_transform, _component_parameters, sample_mixture


def expected_component(model, aifs, gfs):
    m1, s1, m2, s2 = _component_parameters(
        model,
        float(aifs),
        float(gfs),
    )

    w1 = model.weight_aifs
    w2 = model.weight_gfs

    if model.variable in ("tmax", "tmin"):
        return float(w1 * m1 + w2 * m2)

    return max(
        0.0,
        float(
            w1 * (math.exp(m1 + 0.5 * s1 * s1) - 1.0)
            + w2 * (math.exp(m2 + 0.5 * s2 * s2) - 1.0)
        ),
    )

def temperature_samples(models, location, a_tmax,g_tmax,a_tmin,g_tmin,n=60000,seed=1234):
    rng = np.random.default_rng(seed)
    s1 = sample_mixture(models["tmax"][location],a_tmax,g_tmax,n,int(rng.integers(0,2**32-1)))
    s2 = sample_mixture(models["tmin"][location],a_tmin,g_tmin,n,int(rng.integers(0,2**32-1)))
    return 0.5*(s1+s2)


def load_temperature_eval(aifs_dir,gfs_dir,imd_dir,start,end):
    a=records_to_frame(build_daily_extremes(aifs_dir,start_date=str(start.date()),end_date=str(end.date()),min_timesteps=4))
    g=records_to_frame(build_daily_extremes(gfs_dir,start_date=str(start.date()),end_date=str(end.date()),min_timesteps=4))
    a=a.rename(columns={"tmax":"aifs_tmax","tmin":"aifs_tmin"})
    g=g.rename(columns={"tmax":"gfs_tmax","tmin":"gfs_tmin"})
    df=a.merge(g,on=["target_date","location"],how="inner")
    obs=load_imd_series(imd_dir,"temperature",df.target_date.tolist())
    df["obs_tmax"]=[obs.get((r.location,r.target_date,"tmax"),np.nan) for r in df.itertuples()]
    df["obs_tmin"]=[obs.get((r.location,r.target_date,"tmin"),np.nan) for r in df.itertuples()]
    return df.dropna(subset=["aifs_tmax","gfs_tmax","aifs_tmin","gfs_tmin","obs_tmax","obs_tmin"])


def evaluate_temperature(models,df):
    out=[]
    for loc in LOCATIONS:
        sub=df[df.location==loc]
        if sub.empty:
            out.append({"variable":"temperature","location":loc,"status":"NO_EVAL_SAMPLES"}); continue
        obs=0.5*(sub.obs_tmax.to_numpy(float)+sub.obs_tmin.to_numpy(float))
        a_point=0.5*(sub.aifs_tmax.to_numpy(float)+sub.aifs_tmin.to_numpy(float))
        g_point=0.5*(sub.gfs_tmax.to_numpy(float)+sub.gfs_tmin.to_numpy(float))
        b_pred=[]; lows=[]; highs=[]
        for r in sub.itertuples():
            b_pred.append(0.5*(expected_component(models['tmax'][loc],r.aifs_tmax,r.gfs_tmax)+expected_component(models['tmin'][loc],r.aifs_tmin,r.gfs_tmin)))
            smp=temperature_samples(models,loc,r.aifs_tmax,r.gfs_tmax,r.aifs_tmin,r.gfs_tmin,n=30000,seed=hash((loc,str(r.target_date))) & 0xffffffff)
            lows.append(float(np.quantile(smp,.05))); highs.append(float(np.quantile(smp,.95)))
        b=np.asarray(b_pred); lo=np.asarray(lows); hi=np.asarray(highs); e=b-obs
        out.append({"variable":"temperature","location":loc,"n":int(len(obs)),"aifs_mae":float(np.mean(np.abs(a_point-obs))),"gfs_mae":float(np.mean(np.abs(g_point-obs))),"bma_mae":float(np.mean(np.abs(e))),"bma_rmse":float(np.sqrt(np.mean(e*e))),"bma_bias":float(np.mean(e)),"bma_within_tolerance_pct":float(100*np.mean(np.abs(e)<=2.0)),"bma_90_coverage_pct":float(100*np.mean((obs>=lo)&(obs<=hi))),"bma_90_mean_width":float(np.mean(hi-lo)),"status":"OK"})
    return out


def evaluate_precip(models,df):
    out=[]
    for loc in LOCATIONS:
        sub=df[df.location==loc]
        if sub.empty:
            out.append({"variable":"precipitation","location":loc,"status":"NO_EVAL_SAMPLES"}); continue
        y=sub.observed.to_numpy(float); a=sub.aifs.to_numpy(float); g=sub.gfs.to_numpy(float); m=models['precipitation'][loc]
        pred=np.array([expected_component(m,x,z) for x,z in zip(a,g)])
        intervals=[]
        for x,z in zip(a,g):
            sm=sample_mixture(m,x,z,40000,12345); intervals.append((float(np.quantile(sm,.05)),float(np.quantile(sm,.95))))
        intervals=np.asarray(intervals); e=pred-y
        out.append({"variable":"precipitation","location":loc,"n":int(len(y)),"aifs_mae":float(np.mean(np.abs(a-y))),"gfs_mae":float(np.mean(np.abs(g-y))),"bma_mae":float(np.mean(np.abs(e))),"bma_rmse":float(np.sqrt(np.mean(e*e))),"bma_bias":float(np.mean(e)),"bma_within_tolerance_pct":float(100*np.mean(np.abs(e)<=5.0)),"bma_90_coverage_pct":float(100*np.mean((y>=intervals[:,0])&(y<=intervals[:,1]))),"bma_90_mean_width":float(np.mean(intervals[:,1]-intervals[:,0])),"status":"OK"})
    return out


def print_rows(rows):
    print(f"\n=== {rows[0]['variable'].upper()} BMA EVALUATION ===")
    tol=2 if rows[0]['variable']=='temperature' else 5
    unit='°C' if tol==2 else 'mm'
    print(f"Within-tolerance threshold: ±{tol} {unit}")
    print(f"{'Location':<12}{'N':>6}{'AIFS MAE':>11}{'GFS MAE':>11}{'BMA MAE':>11}{'BMA RMSE':>11}{'Bias':>10}{'Acc* %':>10}{'90% Cov':>10}")
    print('-'*92)
    for r in rows:
        if r['status']!='OK': print(f"{r['location']:<12}{r['status']:>12}"); continue
        print(f"{r['location']:<12}{r['n']:>6}{r['aifs_mae']:>11.3f}{r['gfs_mae']:>11.3f}{r['bma_mae']:>11.3f}{r['bma_rmse']:>11.3f}{r['bma_bias']:>10.3f}{r['bma_within_tolerance_pct']:>10.1f}{r['bma_90_coverage_pct']:>10.1f}")
    print('* Acc% = percentage within chosen tolerance; 90% Cov = empirical coverage of the BMA central 90% interval.')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--model-file',default='./models/bma_models_2025.json'); p.add_argument('--aifs-dir',default='./data/regridded/aifs_2025'); p.add_argument('--gfs-dir',default='./data/regridded/gfs_2025'); p.add_argument('--imd-dir',default='./data/imd_2025'); p.add_argument('--variable',choices=VARIABLES+('all',),default='all'); p.add_argument('--start',default='2025-10-01'); p.add_argument('--end',default='2025-12-31'); p.add_argument('--csv'); p.add_argument('--json',dest='json_path'); a=p.parse_args()
    start=pd.Timestamp(a.start).normalize(); end=pd.Timestamp(a.end).normalize()
    models=load_models(Path(a.model_file).resolve()); results=[]
    if a.variable in ('temperature','all'):
        rows=evaluate_temperature(models,load_temperature_eval(Path(a.aifs_dir).resolve(),Path(a.gfs_dir).resolve(),Path(a.imd_dir).resolve(),start,end)); results+=rows; print_rows(rows)
    if a.variable in ('precipitation','all'):
        adf = load_precip_forecast_records(
            Path(a.aifs_dir).resolve(),
            'AIFS'
        )
        gdf = load_precip_forecast_records(
            Path(a.gfs_dir).resolve(),
            'GFS'
        )

        af = adf.rename(columns={'precipitation': 'aifs'})[
            ['target_date', 'location', 'aifs']
        ]

        gf = gdf.rename(columns={'precipitation': 'gfs'})[
            ['target_date', 'location', 'gfs']
        ]

        df = af.merge(
            gf,
            on=['target_date', 'location'],
            how='inner'
        )

        df = df[
            (df.target_date >= start) &
            (df.target_date <= end)
        ]

        obs = load_imd_series(
            Path(a.imd_dir).resolve(),
            'precipitation',
            df.target_date.tolist()
        )

        df['observed'] = [
            obs.get((r.location, r.target_date), np.nan)
            for r in df.itertuples()
        ]

        df = df.dropna(
            subset=['aifs', 'gfs', 'observed']
        )

        rows = evaluate_precip(models, df)
        results += rows
        print_rows(rows)
    overlap=False
    for var in models:
        for m in models[var].values():
            if start<=pd.Timestamp(m.training_end) and end>=pd.Timestamp(m.training_start): overlap=True
    if overlap: print('\nWARNING: evaluation period overlaps saved model training period; these are not held-out test metrics.')
    payload={'evaluation_period':[str(start.date()),str(end.date())],'results':results,'training_overlap_warning':overlap}
    if a.csv:
        path=Path(a.csv).resolve(); path.parent.mkdir(parents=True,exist_ok=True); fields=sorted({k for r in results for k in r});
        with path.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(results)
    if a.json_path: Path(a.json_path).resolve().write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return 0
if __name__=='__main__': raise SystemExit(main())
