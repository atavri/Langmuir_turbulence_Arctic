#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 17:34:04 2025

@author: atavri
"""

# ===================================================
# ONE-CELL PIPELINE: NextSIM + WW3 + MLD + MERGED_ALL
# ===================================================

import os, re
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from collections import Counter

YEAR = 2018  # <<—— change this ONLY if you want another year later

# ===============================
# 1) LOAD NEXTSIM (2018)
# ===============================
def load_nextsim_data_2018(ns_data_dir, chunks=None):
    ns_dict = {}
    all_ts = []

    subdirectories = [f.path for f in os.scandir(ns_data_dir) if f.is_dir()]
    print("NextSIM subdirectories:", len(subdirectories))

    pat1 = re.compile(rf"Moorings_{YEAR}(\d{{2}})\.nc")
    pat2 = re.compile(rf"Moorings_{YEAR}m(\d{{2}})\.nc")

    for subdirectory in subdirectories:
        for nc_file in Path(subdirectory).glob(f"**/Moorings_{YEAR}*.nc"):
            if not (pat1.search(nc_file.name) or pat2.search(nc_file.name)):
                continue

            try:
                ds = xr.open_dataset(nc_file, decode_times=True, chunks=chunks)[
                    ["latitude","longitude","time","sic","sit","taux","tauy",
                     "siu","siv","dmean","dmax","sic_young","sit_young",
                     "sst","sss","t2m"]
                ]
            except Exception as e:
                print(f"Could not open {nc_file}: {e}")
                continue

            full_timestamps = pd.to_datetime(ds["time"].values)
            mask = full_timestamps.year == YEAR
            if not mask.any(): continue

            ds = ds.isel(time=mask)
            print(f"{nc_file.name}: {mask.sum()} timestamps")

            for i, ts in enumerate(full_timestamps[mask]):
                key = pd.Timestamp(ts).strftime("%Y%m%dT%H%M%S")
                ns_dict[key] = ds.isel(time=i)

            all_ts.extend(full_timestamps[mask])

    base_timestamp_strs = sorted({pd.Timestamp(t).strftime("%Y%m%dT%H%M%S") 
                                  for t in all_ts})
    return ns_dict, base_timestamp_strs


NS_DATA_DIR = Path("/oscar/data/deeps/private/chorvat/atavri/Langmuir_turbulence_coupled_model/Coupled_NextSim_WW3_sims/nextsim_outputs")
ns_dict, ns_base_dates = load_nextsim_data_2018(NS_DATA_DIR, chunks={"time": 64})


# ===============================
# 2) LOAD WW3 (2018)
# ===============================
def load_ww3_data_2018(ww3_data_dir, keep_keys=None, chunks=None):

    ATTR_MAP = {
        "uss": ["uuss","vuss"],
        "wnd": ["uwnd","vwnd"],
        "fp":  ["fp"],
        "hs":  ["hs"],
        "tus": ["tus"],
        "dp":  ["dp"],
        "dir": ["dir"],
        "ice": ["ice"],
    }
    WANTED = set(sum(ATTR_MAP.values(), []))

    ww3_dict = {}
    loaded_vars = set()
    keep = set(keep_keys) if keep_keys else None
    subdirs = [f.path for f in os.scandir(ww3_data_dir) if f.is_dir()]
    print("WW3 subdirectories:", len(subdirs))

    pat_y  = re.compile(rf"ww3\.{YEAR}_(\w+)\.nc$", re.IGNORECASE)
    pat_ym = re.compile(rf"ww3\.{YEAR}\d{{2}}_(\w+)\.nc$", re.IGNORECASE)

    def _open_safely(nc_path, chunks):
        for drop in (["longitude","latitude"],["lon","lat"],["longitude","latitude","lon","lat"]):
            try:
                return xr.open_dataset(
                    nc_path, engine="netcdf4", decode_times=True,
                    decode_coords=False, mask_and_scale=True,
                    drop_variables=drop, chunks=chunks
                )
            except Exception:
                pass
        return xr.open_dataset(
            nc_path, engine="netcdf4", decode_cf=False,
            drop_variables=["latitude","longitude","lon","lat"],
            chunks=chunks
        )

    def find_vars_ci(ds, names):
        output = []
        lower_map = {k.lower(): k for k in list(ds.data_vars) + list(ds.coords)}
        for n in names:
            if n.lower() in lower_map:
                output.append(lower_map[n.lower()])
        return output

    counts = Counter()

    for subdir in subdirs:
        for nc_file in Path(subdir).glob(f"**/ww3.{YEAR}*.nc"):
            fname = nc_file.name
            m = pat_ym.search(fname) or pat_y.search(fname)
            if not m: continue

            attr = m.groups()[0].lower()
            if attr not in ATTR_MAP: continue

            try:
                ds = _open_safely(nc_file, chunks=chunks)
                ren = {}
                if "latitude" in ds.dims: ren["latitude"] = "y"
                if "longitude" in ds.dims: ren["longitude"] = "x"
                if ren: ds = ds.rename(ren)
            except:
                print("Could not open", fname)
                continue

            if "time" not in ds: continue

            desired = find_vars_ci(ds, ATTR_MAP[attr])
            if not desired:
                dv = [v for v in ds.data_vars if v not in ("time","y","x")]
                if len(dv) == 1: desired = [dv[0]]

            times = pd.to_datetime(ds["time"].values)
            mask = (times.year == YEAR)
            if not mask.any(): continue
            times = times[mask]
            ds = ds.isel(time=mask)

            for i, ts in enumerate(times):
                key = pd.Timestamp(ts).strftime("%Y%m%dT%H%M%S")
                if keep and key not in keep: continue
                ww3_dict.setdefault(key, {})

                for var in desired:
                    try:
                        sel = ds[var].isel(time=i)
                        store = var.lower() if var.lower() in WANTED else var
                        ww3_dict[key][store] = sel
                        loaded_vars.add(store)
                    except:
                        pass

                counts[YEAR] += 1

    print("WW3 timestamps loaded:", counts[YEAR])
    return ww3_dict, sorted(loaded_vars)


WW3_DATA_DIR = Path("/oscar/data/deeps/private/chorvat/atavri/Langmuir_turbulence_coupled_model/Coupled_NextSim_WW3_sims/ww3_outputs")
ww3_dict, ww3_vars = load_ww3_data_2018(WW3_DATA_DIR, chunks={"time": 64})
print("WW3 vars:", ww3_vars)


# ===============================
# 3) FIND COMMON TIMESTAMPS
# ===============================
common_timestamps = sorted(set(ns_dict.keys()) & set(ww3_dict.keys()))
print("Common timestamps:", len(common_timestamps))

ns_dict_common  = {k: ns_dict[k] for k in common_timestamps}
ww3_dict_common = {k: ww3_dict[k] for k in common_timestamps}


# ===============================
# 4) ALIGN WW3 GRID → NextSIM GRID
# ===============================
def get_ns_latlon_2d(ds):
    lat = ds["latitude"]
    lon = ds["longitude"]
    if "time" in lat.dims: lat = lat.isel(time=0)
    if "time" in lon.dims: lon = lon.isel(time=0)
    if lat.ndim != 2:
        lat = lat.squeeze()
        lon = lon.squeeze()
    ren = {}
    if "latitude" in lat.dims: ren["latitude"] = "y"
    if "longitude" in lon.dims: ren["longitude"] = "x"
    if ren:
        lat = lat.rename(ren)
        lon = lon.rename(ren)
    return lat.astype("float32"), lon.astype("float32")

print("Aligning WW3 → NextSIM grid ...")
ok, fail = 0, 0

for ts in common_timestamps:
    lat2d, lon2d = get_ns_latlon_2d(ns_dict_common[ts])

    for var, w in ww3_dict_common[ts].items():
        try:
            if "latitude" in w.coords:  w = w.drop_vars("latitude")
            if "longitude" in w.coords: w = w.drop_vars("longitude")

            if (w.sizes["y"] != lat2d.sizes["y"]) or (w.sizes["x"] != lon2d.sizes["x"]):
                raise ValueError("grid mismatch")

            ww3_dict_common[ts][var] = w.assign_coords(latitude=lat2d, longitude=lon2d)
            ok += 1
        except Exception as e:
            print("⚠️", ts, var, e)
            fail += 1

print(f"Done: aligned OK={ok}, FAIL={fail}")

#%%
# ===============================
# 5) LOAD + MATCH MLD
# ===============================
MLD_FILE = Path("/oscar/data/deeps/private/chorvat/atavri/Langmuir_turbulence_coupled_model/Coupled_NextSim_WW3_sims/nextsim_outputs/MLD_daily.nc")

ds_mld = xr.open_dataset(
    MLD_FILE, engine="netcdf4", chunks={"time":1},
    decode_times=False, mask_and_scale=False
)

if not np.issubdtype(ds_mld["time"].dtype, np.datetime64):
    if "units" in ds_mld["time"].attrs:
        ds_mld = xr.decode_cf(ds_mld)
    else:
        ds_mld = ds_mld.assign_coords(
            time=pd.date_range(f"{YEAR}-01-01",
                               periods=ds_mld.sizes["time"], freq="3H")
        )

ds_mld_year = ds_mld.sel(time=ds_mld.time.dt.year == YEAR)

common_dt = pd.to_datetime(common_timestamps, format="%Y%m%dT%H%M%S")
common_days = np.unique(common_dt.normalize())

mld_days = pd.to_datetime(ds_mld_year.time.values).normalize()
mask = np.isin(mld_days, common_days)

ds_mld_common = ds_mld_year.isel(time=mask)
print("MLD matched:", ds_mld_common.sizes["time"])


# ===============================
# 6) MERGE ALL INTO ONE dataset
# ===============================
merged_datasets = []

for key in common_timestamps:
    try:
        ts = pd.to_datetime(key, format="%Y%m%dT%H%M%S")

        ns_ds = ns_dict_common[key]
        ww3_ds = xr.Dataset(ww3_dict_common[key])
        mld_ds = ds_mld_common.sel(time=ts, method="nearest")

        ds_merged = xr.merge([ns_ds, ww3_ds, mld_ds],
                             compat="override", join="inner")
        ds_merged = ds_merged.expand_dims(time=[np.datetime64(ts)])

        merged_datasets.append(ds_merged)
    except Exception as e:
        print("⚠️ Skipping", key, e)

merged_all = xr.concat(merged_datasets, dim="time").sortby("time")
merged_all = merged_all.chunk({"time": 24, "y": 256, "x": 256})

print("\n======== FINAL MERGED DATASET ========")
print(merged_all)
print("Variables:", list(merged_all.data_vars))
#%%
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
import time
from pathlib import Path

# --------------------------------------------
# Physical constants
# --------------------------------------------
RHO = 1025
EPS = 1e-8
KAPPA = 0.4
Z1 = 1.0
XI = 0.5
G = 9.81
C1 = 3.1
C2 = 5.7

# --------------------------------------------
# Helper
# --------------------------------------------
def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

# --------------------------------------------
# α_LOW (Van Roekel 2012) — FIXED VERSION
# --------------------------------------------
def alpha_low(ustar, us0, theta, DL):

    us0_s = xr.where(us0 > EPS, us0, EPS)

    Ss = us0_s / DL
    d_us_dx = Ss * np.cos(theta)
    d_us_dy = Ss * np.sin(theta)

    denom = (ustar / KAPPA) * np.log(DL / Z1) + d_us_dx

    ang = np.arctan2(d_us_dy, denom)
    return wrap_pi(ang)

# --------------------------------------------
# MAIN Langmuir calculator — FIXED VERSION
# --------------------------------------------
def compute_langmuir(ds):

    print("→ Computing Langmuir diagnostics ...")

    # Peak frequency
    fp_lin = 10.0 ** ds.fp
    kp = ((2*np.pi*fp_lin)**2) / G

    # Production depth
    H = xr.where(ds.MLD > Z1, ds.MLD, Z1 + 1e-6)

    DL_wave = XI / xr.where(kp > 0, kp, np.nan)
    DL_mld = 0.2 * H

    # FIXED: replace ufuncs.minimum with np.minimum
    DL = xr.where(np.isfinite(DL_wave), np.minimum(DL_wave, DL_mld), DL_mld)
    DL = DL.clip(min=1.0)

    # Momentum forcing
    tau = np.hypot(ds.taux, ds.tauy)
    ustar = xr.where(tau > 0, np.sqrt(tau / RHO), EPS)

    us0 = np.hypot(ds.uuss, ds.vuss)
    us0 = xr.where(us0 > EPS, us0, EPS)

    # Angles
    wind_angle = np.arctan2(ds.vwnd, ds.uwnd)
    stokes_angle = np.arctan2(ds.vuss, ds.uuss)
    theta = wrap_pi(wind_angle - stokes_angle)

    # Floors
    ustar_s = xr.where(ustar > EPS, ustar, EPS)
    us0_s = xr.where(us0 > EPS, us0, EPS)

    # α_LOW
    aL = alpha_low(ustar_s, us0_s, theta, DL)

    cos_ustar = np.abs(np.cos(aL)).clip(min=0.01)
    cos_us = np.abs(np.cos(theta - aL)).clip(min=0.01)

    # Langmuir numbers
    La_t = np.sqrt(ustar_s / us0_s).clip(0, 200)
    La_proj_LOW = np.sqrt((ustar_s * cos_ustar) / (us0_s * cos_us)).clip(0, 200)

    # Energetics
    E_La = np.sqrt(1 + (C1*La_t)**-2 + (C2*La_t)**-4)

    eps_shear = (ustar_s**3) / H
    eps_total_vr = eps_shear * E_La

    # Output
    ds_out = xr.Dataset(
        {
            "DL": DL,
            "fp_lin": fp_lin,
            "kp": kp,
            "alpha_LOW": aL,
            "La_t": La_t,
            "La_proj_LOW": La_proj_LOW,
            "eps_shear": eps_shear,
            "eps_total_vr": eps_total_vr,
        },
        coords={
            "time": ds.time,
            "y": ds.y,
            "x": ds.x,
            "latitude": ds.latitude,
            "longitude": ds.longitude,
        }
    )

    print("✔ Langmuir computed.")
    return ds_out


# --------------------------------------------
# Save wrapper
# --------------------------------------------
def compute_and_save(ds_all, out_path):

    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    print(f"\n=== Computing + Saving Langmuir file ===")
    print(f"→ Output: {out_path}")

    t0 = time.time()

    result = compute_langmuir(ds_all)

    encoding = {
        v: {"zlib": True, "complevel": 2, "dtype": "float32"}
        for v in result.data_vars
    }

    with ProgressBar():
        result.to_netcdf(out_path, engine="netcdf4", encoding=encoding)

    print(f"✔ Saved {out_path}")
    print(f"Runtime: {(time.time() - t0)/60:.2f} min")

    return out_path
#
compute_and_save(merged_all, "outputs/Langmuir_2018_fixed.nc")