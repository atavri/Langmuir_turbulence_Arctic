#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 18:50:43 2025

@author: atavri
"""

#test run for 2018
# --- Cell 2: NextSIM loader (2018 only) ---
import os, re
from pathlib import Path
import xarray as xr
import pandas as pd

def load_nextsim_data_2018(ns_data_dir, chunks=None):
    """
    Load only 2018 NextSIM data (Moorings_2018*.nc).
    Scans subfolders for matching files and builds a dict keyed by 'YYYYMMDDTHHMMSS'.
    Returns (ns_dict, base_date_strs_sorted)
    """
    ns_dict = {}
    all_ts = []

    subdirectories = [f.path for f in os.scandir(ns_data_dir) if f.is_dir()]
    print("NextSIM subdirectories:", len(subdirectories))

    # Match 2018 files only
    pat1 = re.compile(r"Moorings_2018(\d{2})\.nc")     # e.g., Moorings_201808.nc
    pat2 = re.compile(r"Moorings_2018m(\d{2})\.nc")    # e.g., Moorings_2018m08.nc

    for subdirectory in subdirectories:
        for nc_file in Path(subdirectory).glob("**/Moorings_2018*.nc"):
            if not (pat1.search(nc_file.name) or pat2.search(nc_file.name)):
                continue

            try:
                ds = xr.open_dataset(nc_file, decode_times=True, chunks=chunks)[
                    ["latitude", "longitude", "time", "sic", "sit", "taux", "tauy",
                     "siu", "siv", "dmean", "dmax", "sic_young", "sit_young",
                     "sst", "sss", "t2m"]
                ]
            except Exception as e:
                print(f"Could not open {nc_file}: {e}")
                continue

            # Filter timestamps to 2018 just in case file overlaps
            full_timestamps = pd.to_datetime(ds["time"].values)
            mask_2018 = (full_timestamps.year == 2018)
            if not mask_2018.any():
                continue
            ds = ds.isel(time=mask_2018)

            print(f"{nc_file.name}: {mask_2018.sum()} timestamps (2018 only)")

            for i, ts in enumerate(full_timestamps[mask_2018]):
                key = pd.Timestamp(ts).strftime('%Y%m%dT%H%M%S')
                ns_dict[key] = ds.isel(time=i)
            all_ts.extend(full_timestamps[mask_2018])

    base_date_strs = sorted({pd.Timestamp(t).strftime('%Y%m%dT%H%M%S') for t in all_ts})
    return ns_dict, base_date_strs
#%%
from pathlib import Path
NS_DATA_DIR = Path("/Users/atavri/analysis/nextsim_outputs")
ns_dict, base_date_strs = load_nextsim_data_2018(NS_DATA_DIR, chunks={"time": 64})

#%%
# --- Cell: WW3 loader (2018 only) ---
import os, re
from pathlib import Path
from collections import Counter
import pandas as pd
import xarray as xr

def load_ww3_data_2018(ww3_data_dir, keep_keys=None, chunks=None):
    """
    Load only 2018 WW3 data (files and timestamps).
    Returns:
        ww3_dict: {'YYYYMMDDTHHMMSS': {var_name: xr.DataArray}}
        ww3_vars: sorted list of actually-loaded variable names
    """
    # variable mapping
    ATTR_MAP = {
        "uss": ["uuss", "vuss"],
        "wnd": ["uwnd", "vwnd"],
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
    print("Found WW3 subdirectories:", len(subdirs))

    # patterns (accepts ww3.2018_attr.nc or ww3.201801_attr.nc)
    pat_y  = re.compile(r"ww3\.2018_(\w+)\.nc$", re.IGNORECASE)
    pat_ym = re.compile(r"ww3\.2018\d{2}_(\w+)\.nc$", re.IGNORECASE)

    def _open_safely(nc_path, chunks):
        for drop_vars in (["longitude","latitude"], ["lon","lat"], ["longitude","latitude","lon","lat"]):
            try:
                return xr.open_dataset(
                    nc_path,
                    engine="netcdf4",
                    decode_times=True,
                    decode_coords=False,
                    mask_and_scale=True,
                    chunks=chunks,
                    drop_variables=drop_vars,
                )
            except Exception:
                continue
        # fallback
        return xr.open_dataset(
            nc_path,
            engine="netcdf4",
            decode_cf=False,
            chunks=chunks,
            drop_variables=["longitude","latitude","lon","lat"],
        )

    def find_vars_ci(ds, names):
        out = []
        lower_map = {k.lower(): k for k in list(ds.data_vars) + list(ds.coords)}
        for name in names:
            k = lower_map.get(name.lower())
            if k is not None:
                out.append(k)
        return out

    counts_by_year = Counter()

    for subdir in subdirs:
        for nc_file in Path(subdir).glob("**/ww3.2018*.nc"):
            fname = nc_file.name
            m = pat_ym.search(fname) or pat_y.search(fname)
            if not m:
                continue
            attribute = m.groups()[0].lower()
            if attribute not in ATTR_MAP:
                continue

            try:
                ds = _open_safely(nc_file, chunks=chunks)
                dim_ren = {}
                if "latitude" in ds.dims: dim_ren["latitude"] = "y"
                if "longitude" in ds.dims: dim_ren["longitude"] = "x"
                if dim_ren: ds = ds.rename(dim_ren)
            except Exception as e:
                print(f"Could not open {fname}: {e}")
                continue

            if "time" not in ds: continue
            desired = find_vars_ci(ds, ATTR_MAP[attribute])
            if not desired:
                data_vars = [v for v in ds.data_vars if v not in ("time","y","x","latitude","longitude")]
                if len(data_vars) == 1:
                    desired = [data_vars[0]]

            times = pd.to_datetime(ds["time"].values)
            mask_2018 = times.year == 2018
            if not mask_2018.any():
                continue
            times = times[mask_2018]
            ds = ds.isel(time=mask_2018)

            for i, ts in enumerate(times):
                key = pd.Timestamp(ts).strftime('%Y%m%dT%H%M%S')
                if keep and key not in keep:
                    continue
                ww3_dict.setdefault(key, {})
                for var in desired:
                    try:
                        sel = ds[var].isel(time=i)
                        store_name = var.lower() if var.lower() in WANTED else var
                        ww3_dict[key][store_name] = sel
                        loaded_vars.add(store_name)
                    except Exception:
                        pass
                counts_by_year[2018] += 1

    if counts_by_year:
        print("WW3 timestamps (2018):", counts_by_year[2018])
    else:
        print("No WW3 timestamps found for 2018.")
    return ww3_dict, sorted(loaded_vars)
#%%
from pathlib import Path

# Define your WW3 data directory
WW3_DATA_DIR = Path("/Users/atavri/Desktop/analysis/WW3_outputs")

# Run the loader
ww3_dict, ww3_vars = load_ww3_data_2018(WW3_DATA_DIR, chunks={"time": 64})
print("Loaded WW3 variables:", ww3_vars)
#%%
# --- Cell: Find common 2018 timestamps between NextSIM and WW3 ---

# Ensure both dictionaries exist from previous cells
if 'ns_dict' not in locals() or 'ww3_dict' not in locals():
    raise RuntimeError("Please run the loaders first (ns_dict and ww3_dict must be defined).")

# Convert timestamp keys to sets
ns_timestamps = set(ns_dict.keys())
ww3_timestamps = set(ww3_dict.keys())

# Intersection: timestamps present in both datasets
common_timestamps = sorted(ns_timestamps & ww3_timestamps)

print(f"Common timestamps (2018): {len(common_timestamps)}")
if common_timestamps:
    print("First few:", common_timestamps[:5])
else:
    print("No overlapping timestamps found between NextSIM and WW3 for 2018.")

# Keep only shared timestamps
ns_dict_common = {k: ns_dict[k] for k in common_timestamps}
ww3_dict_common = {k: ww3_dict[k] for k in common_timestamps}

print(f"ns_dict_common: {len(ns_dict_common)}  |  ww3_dict_common: {len(ww3_dict_common)}")

# (optional) Verify a sample key
if common_timestamps:
    key0 = common_timestamps[0]
    print(f"\nSample timestamp: {key0}")
    print("NextSIM vars:", list(ns_dict_common[key0].data_vars))
    print("WW3 vars:", list(ww3_dict_common[key0].keys()))

#%%
# --- Cell 5: Align WW3 coordinates to NextSIM grid (2018 only) ---
import numpy as np
import xarray as xr

def get_ns_latlon_2d(ns_ds):
    """Extract 2D latitude/longitude from a NextSIM Dataset (handles (time,y,x) or (y,x))."""
    lat = ns_ds["latitude"]
    lon = ns_ds["longitude"]

    # If lat/lon vary with time, take the first slice
    if "time" in lat.dims:
        lat = lat.isel(time=0)
    if "time" in lon.dims:
        lon = lon.isel(time=0)

    # Ensure dims are (y, x)
    if lat.dims != ("y", "x"):
        lat = lat.squeeze()
        lon = lon.squeeze()
        ren = {}
        if "latitude" in lat.dims:
            ren["latitude"] = "y"
        if "longitude" in lat.dims:
            ren["longitude"] = "x"
        if ren:
            lat = lat.rename(ren)
            lon = lon.rename(ren)

    # Convert to float32 for memory efficiency
    if str(lat.dtype).startswith("float64"):
        lat = lat.astype("float32")
    if str(lon.dtype).startswith("float64"):
        lon = lon.astype("float32")

    return lat, lon


# --- Align all WW3 slices to the 2018 NextSIM grid ---
n_align_ok = 0
n_align_fail = 0

# Make sure the required dictionaries exist
if 'ns_dict_common' not in locals() or 'ww3_dict_common' not in locals() or 'common_timestamps' not in locals():
    raise RuntimeError("Please run the previous cells (loaders and timestamp matching) first.")

print(f"Aligning WW3 coordinates to NextSIM grid for {len(common_timestamps)} common 2018 timestamps...")

for ts_str in common_timestamps:
    ns_ds = ns_dict_common[ts_str]
    lat2d, lon2d = get_ns_latlon_2d(ns_ds)

    for varname, ww3_slice in ww3_dict_common[ts_str].items():
        try:
            # Drop old coordinates if present
            if "latitude" in ww3_slice.coords:
                ww3_slice = ww3_slice.drop_vars("latitude")
            if "longitude" in ww3_slice.coords:
                ww3_slice = ww3_slice.drop_vars("longitude")

            # Shape consistency check
            if (ww3_slice.sizes.get("y") != lat2d.sizes.get("y")) or \
               (ww3_slice.sizes.get("x") != lat2d.sizes.get("x")):
                raise ValueError(f"shape mismatch for {varname}: WW3 {ww3_slice.sizes} vs NS {lat2d.sizes}")

            # Reassign aligned coordinates
            ww3_dict_common[ts_str][varname] = ww3_slice.assign_coords(latitude=lat2d, longitude=lon2d)
            n_align_ok += 1

        except Exception as e:
            print(f"align error @ {ts_str} for '{varname}': {e}")
            n_align_fail += 1

print(f"\n WW3 latitude/longitude aligned to NextSIM grid (2018 only): ok={n_align_ok}, fail={n_align_fail}")
#%%
# --- Cell: Load and prepare MLD dataset (2018 only) ---
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

# Path to your MLD file
MLD_FILE = Path("./MLD_daily.nc")

# --- Load lazily for performance ---
ds_mld = xr.open_dataset(
    MLD_FILE,
    engine="netcdf4",
    chunks={"time": 1},   # one time per chunk
    decode_times=False,
    mask_and_scale=False
)
print("Opened MLD dataset.")
print("Variables:", list(ds_mld.data_vars))
print("Dimensions:", ds_mld.dims)

# --- Decode time correctly ---
if not np.issubdtype(ds_mld["time"].dtype, np.datetime64):
    if "units" in ds_mld["time"].attrs:
        ds_mld = xr.decode_cf(ds_mld)
    else:
        # Fallback: assume 3-hourly data starting Jan 1, 2018
        ds_mld = ds_mld.assign_coords(
            time=pd.date_range("2018-01-01", periods=ds_mld.sizes["time"], freq="3H")
        )

print("Time decoded:", ds_mld.time.values[0], "→", ds_mld.time.values[-1])

# --- Filter to 2018 ---
ds_mld_2018 = ds_mld.sel(time=ds_mld.time.dt.year == 2018)
print("Filtered to 2018:", ds_mld_2018.time.size, "timesteps")

# --- Match to your model timestamps ---
# Convert model timestamps (strings) to datetime64
common_dt = pd.to_datetime(common_timestamps, format="%Y%m%dT%H%M%S")
common_days = np.unique(common_dt.normalize())  # daily unique timestamps

# Normalize MLD timestamps to daily values
mld_days = pd.to_datetime(ds_mld_2018.time.values).normalize()
mask = np.isin(mld_days, common_days)

# Keep only matching times
ds_mld_common = ds_mld_2018.isel(time=mask)
print(f"MLD overlapping timesteps with model data: {ds_mld_common.sizes['time']}")

# --- Final sanity check ---
print(ds_mld_common)

#%%
import numpy as np
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar

print(f" Merging {len(common_timestamps)} timestamps (2018) from NextSIM, WW3, and MLD...")

merged_datasets = []

for key in common_timestamps:
    try:
        # Convert timestamp string (e.g., '20180101T000000') to datetime64
        ts = pd.to_datetime(key, format="%Y%m%dT%H%M%S")

        # --- Load datasets for this timestamp ---
        ns_ds = ns_dict_common[key]
        ww3_ds = xr.Dataset(ww3_dict_common[key])
        mld_ds = ds_mld_common.sel(time=ts, method="nearest")

        # --- Merge NextSIM + WW3 + MLD ---
        ds_merged = xr.merge([ns_ds, ww3_ds, mld_ds], compat="override", join="inner")

        # --- Add time dimension (expand to 4D -> (time, y, x)) ---
        ds_merged = ds_merged.expand_dims(time=[np.datetime64(ts)])

        merged_datasets.append(ds_merged)

    except Exception as e:
        print(f"Skipping {key}: {e}")
        continue

print(f"Successfully merged {len(merged_datasets)} time steps.")

# --- Combine all into one dataset ---
if not merged_datasets:
    raise RuntimeError("No datasets merged successfully — check timestamp consistency.")

merged_all = xr.concat(merged_datasets, dim="time").sortby("time")

# --- Chunk for performance ---
merged_all = merged_all.chunk({"time": 24, "y": 256, "x": 256})

print("\nFinal merged dataset summary:")
print(merged_all)
print("\nVariables:", list(merged_all.data_vars))

#%%

#nneeeewwwww calculationssssssssss

import xarray as xr
import numpy as np
from pathlib import Path
from dask.diagnostics import ProgressBar

# ======================================================
# Assumption: merged_all already exists
#     and contains: taux, tauy, uuss, vuss, uwnd, vwnd, fp, MLD, ...
# ======================================================
# If you want to load from file instead, uncomment and set the path:
# merged_all = xr.open_dataset("/path/to/your/merged_all_2018.nc")

# ======================================================
# Constants and helper functions
# ======================================================
RHO_WATER = 1025.0
EPS       = 1e-8
KAPPA     = 0.4
Z1        = 1.0
G         = 9.81
XI        = 0.5   # Stokes penetration scale (0.5–1 typical)

# Numerical / physical floors
STOKES_MIN = 1e-4   # m/s; below this we treat Stokes drift as "no meaningful LT"
COS_MIN    = 0.05   # minimum |cos| to avoid blow-up under near-90° misalignment

def wrap_pi(a):
    """Wrap angles into [-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi

def compute_alpha_low_fast(ustar, us0, theta_ww, DL, z1=Z1, kappa=KAPPA):
    """
    Compute α_LOW (Law-of-the-Wall Langmuir angle) following Van Roekel et al. (2012).
    ustar, us0, theta_ww, DL are xarray DataArrays.
    """
    eps = 1e-12
    us0_safe = xr.where(us0 > eps, us0, eps)

    # Mean Stokes shear magnitude
    Ss = us0_safe / DL
    d_us_dx = Ss * xr.ufuncs.cos(theta_ww)
    d_us_dy = Ss * xr.ufuncs.sin(theta_ww)

    denom = (ustar / kappa) * xr.ufuncs.log(DL / z1) + d_us_dx
    num   = d_us_dy
    return wrap_pi(xr.ufuncs.arctan2(num, denom))


# ======================================================
# Prepare derived wave + shear inputs
# ======================================================
print("\nComputing wave + shear inputs...")

# Ensure nanosecond precision for time (silences xarray warning)
merged_all = merged_all.assign_coords(time=merged_all.time.astype("datetime64[ns]"))

# Convert fp from log10(Hz) → Hz
fp_lin = 10.0 ** merged_all["fp"]

# Compute wavenumber k_p for peak frequency
kp = ((2.0 * np.pi * fp_lin) ** 2) / G

# Production depth DL = min(0.2*MLD, ξ/kp), with a floor at Z1
H_ML   = xr.where(merged_all["MLD"] > Z1, merged_all["MLD"], Z1 + 1e-6)
DL_mld = 0.2 * H_ML
DL_wave = XI / xr.where(kp > 0, kp, np.nan)
DL_comb = xr.where(np.isfinite(DL_wave), xr.ufuncs.minimum(DL_mld, DL_wave), DL_mld)
DL      = xr.where(DL_comb > Z1, DL_comb, Z1 + 1e-6)

# Stress → friction velocity u*
tau   = np.hypot(merged_all["taux"], merged_all["tauy"])
ustar = np.sqrt(xr.where(tau < 0, 0, tau) / RHO_WATER)

# Surface Stokes drift magnitude
us0 = np.hypot(merged_all["uuss"], merged_all["vuss"])

# Wind–wave angle θ_ww
wind_angle   = np.arctan2(merged_all["vwnd"],  merged_all["uwnd"])
stokes_angle = np.arctan2(merged_all["vuss"],  merged_all["uuss"])
theta_ww     = wrap_pi(wind_angle - stokes_angle)

# Basic numerical floor to avoid zeros
ustar_s = xr.where(ustar > EPS, ustar, EPS)
us0_s   = xr.where(us0   > EPS, us0,   EPS)

print("✔ Core variables computed")


# ======================================================
# Langmuir numbers with α_LOW and physical floors
# ======================================================
print("\nComputing Langmuir numbers (with α_LOW and floors)...")

# α_LOW orientation
alpha_LOW = compute_alpha_low_fast(ustar_s, us0_s, theta_ww, DL)

# Effective Stokes drift magnitude:
# below STOKES_MIN we consider there is essentially no LT forcing;
# we set these locations to NaN so La_t / La_proj_LOW are undefined there.
us0_eff = xr.where(us0_s > STOKES_MIN, us0_s, np.nan)

# Cosines for projections
cos_ustar_raw = np.cos(alpha_LOW)
cos_us_raw    = np.cos(theta_ww - alpha_LOW)
cos_s_raw     = np.cos(theta_ww)

# Absolute values and floors for denominators
cos_ustar_abs = np.abs(cos_ustar_raw)
cos_us_abs    = np.abs(cos_us_raw)
cos_s_abs     = np.abs(cos_s_raw)

cos_ustar_eff = xr.where(cos_ustar_abs > COS_MIN, cos_ustar_abs, COS_MIN)
cos_us_eff    = xr.where(cos_us_abs    > COS_MIN, cos_us_abs,    COS_MIN)
cos_s_eff     = xr.where(cos_s_abs     > COS_MIN, cos_s_abs,     COS_MIN)

# --- Langmuir numbers ---

# Classical turbulent Langmuir number (mask where us0 is too small)
La_t = np.sqrt(ustar_s / us0_eff)

# "Plain" projected number using only θ_ww (not α_LOW)
La_proj = np.sqrt(ustar_s / (us0_eff * cos_s_eff))

# α_LOW-projected Langmuir number
La_proj_LOW = np.sqrt((ustar_s * cos_ustar_eff) /
                      (us0_eff * cos_us_eff))


# Clip to physical ranges
La_t = La_t.clip(min=0.005, max=200)
La_proj = La_proj.clip(min=0.005, max=200)
La_proj_LOW = La_proj_LOW.clip(min=0.005, max=200)


print("✔ Langmuir metrics computed (NaNs where Stokes drift too weak).")


# ======================================================
# Save Langmuir diagnostics
# ======================================================
print("\nSaving Langmuir diagnostics to NetCDF...")

output_dir = Path("./langmuir_outputs_2018")
output_dir.mkdir(parents=True, exist_ok=True)

fn_out = output_dir / "Langmuir_diagnostics_plus_env_DL_2020_cor.nc"
print(f"→ Saving to {fn_out}")

keep_vars = ["sic", "sit", "hs", "dp", "MLD", "siu", "siv", "dmean"]

out_vars = xr.Dataset(
    {
        "ustar":      ustar.astype("float32"),
        "us0":        us0.astype("float32"),
        "theta_ww":   theta_ww.astype("float32"),
        "alpha_LOW":  alpha_LOW.astype("float32"),
        "La_t":       La_t.astype("float32"),
        "La_proj":    La_proj.astype("float32"),
        "La_proj_LOW":La_proj_LOW.astype("float32"),
        "DL":         DL.astype("float32"),
        "kp":         kp.astype("float32"),
        "fp":         fp_lin.astype("float32"),
        **{v: merged_all[v].astype("float32") for v in keep_vars if v in merged_all},
    }
)

out_vars = out_vars.assign_coords(
    latitude = merged_all["latitude"],
    longitude = merged_all["longitude"],
    time = merged_all["time"],
).chunk({"time": 24, "y": 256, "x": 256})

encoding = {v: {"zlib": True, "complevel": 2} for v in out_vars.data_vars}

with ProgressBar():
    out_vars.to_netcdf(fn_out, engine="netcdf4", encoding=encoding)

print(f"✔ Saved Langmuir diagnostics → {fn_out}")

#%%  ----ENERGETICS--------

import numpy as np
import xarray as xr
from pathlib import Path
from dask.diagnostics import ProgressBar

# ======================================================
# CONSTANTS
# ======================================================
RHO_WATER = 1025.0
EPS = 1e-8

# Belcher parameters
A_s = 2.0
A_L = 0.22   # relative LT efficiency in open-ocean LES
BETA_BELCHER = A_L / A_s   # approx. 0.11

# ======================================================
# INPUT FILE
# ======================================================
fn_in = Path("langmuir_outputs_2018/Langmuir_diagnostics_plus_env_DL_2020_cor.nc")

print(f"\nLoading: {fn_in}")
ds = xr.open_dataset(fn_in, chunks={"time": 24, "y": 256, "x": 256})

# ======================================================
# SHEAR DISSIPATION
# ε_shear = u*³ / MLD
# ======================================================
ustar = xr.where(ds.ustar > EPS, ds.ustar, EPS)
MLD   = xr.where(ds.MLD  > 0,    ds.MLD,  np.nan)

eps_shear = (ustar ** 3) / MLD
eps_shear = eps_shear.astype("float32")

# ======================================================
# BELCHER DISSIPATION:
# ε_total = ε_shear * (1 + β / La_t²)
# ======================================================
La_t = xr.where(ds.La_t > EPS, ds.La_t, EPS)

eps_total_belcher = eps_shear * (1 + BETA_BELCHER / (La_t ** 2))
eps_LT_belcher    = eps_total_belcher - eps_shear

# ======================================================
# LT FRACTION:
# f_LT = ε_LT / ε_total
# ======================================================
f_LT = eps_LT_belcher / eps_total_belcher
f_LT = xr.where(np.isfinite(f_LT), f_LT, 0)

# ======================================================
# SAVE OUTPUT
# ======================================================
ene_out = xr.Dataset(
    {
        "eps_shear": eps_shear,
        "eps_total": eps_total_belcher.astype("float32"),
        "eps_LT": eps_LT_belcher.astype("float32"),
        "f_LT": f_LT.astype("float32"),
    },
    coords={
        "time": ds.time,
        "y": ds.y,
        "x": ds.x,
        "latitude": ds.latitude,
        "longitude": ds.longitude,
    },
)

out_file = Path("Langmuir_energetics_2020.nc")
encoding = {v: {"zlib": True, "complevel": 2} for v in ene_out.data_vars}

print(f"\nSaving energetics → {out_file}")

with ProgressBar():
    ene_out.to_netcdf(out_file, engine="netcdf4", encoding=encoding)

print("✔ DONE — Energetics written to:", out_file)
#%%  ---PLOTS CHAGTP SUGEGSTED
#!/usr/bin/env python3
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# =========================================
# Load diagnostics + energetics
# =========================================
ds = xr.open_dataset("langmuir_outputs_2018/Langmuir_diagnostics_plus_env_DL_2020_cor.nc")
ene = xr.open_dataset("Langmuir_energetics_2020.nc")

# =========================================
# Select month
# =========================================
month = "2020-08"
ds_m  = ds.sel(time=slice(month+"-01", month+"-31"))
ene_m = ene.sel(time=slice(month+"-01", month+"-31"))

print("Selected month:", month, "Npoints:", ene_m.f_LT.count().item())

# =========================================
# HISTOGRAMS: La_t vs La_proj_LOW
# =========================================
x = ds_m.La_t.values.ravel()
y = ds_m.La_proj_LOW.values.ravel()

mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]

print("Valid points:", x.size)

if x.size < 10:
    raise ValueError("Insufficient valid points after cleaning; check variable ranges.")

plt.figure(figsize=(9,6))
plt.hist(ds_m.La_t.values.flatten(), bins=60, density=True, alpha=0.6, label=r"$La_t$")
plt.hist(ds_m.La_proj_LOW.values.flatten(), bins=60, density=True, alpha=0.6,
         label=r"$La_{\mathrm{proj,LOW}}$")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Langmuir number"); plt.ylabel("PDF")
plt.legend(); plt.grid(True, which="both", ls="--")
plt.title(f"Histogram: La_t vs La_proj_LOW ({month})")
plt.show()

# =========================================
#  SCATTER: La_proj_LOW vs θ_ww
# =========================================
plt.figure(figsize=(8,6))
plt.scatter(ds_m.theta_ww.values.flatten(),
            ds_m.La_proj_LOW.values.flatten(),
            s=1, alpha=0.3)
plt.yscale("log")
plt.xlabel(r"Wind–wave angle $\theta_{ww}$")
plt.ylabel(r"$La_{\mathrm{proj,LOW}}$")
plt.grid(True, which="both")
plt.title(f"La_proj_LOW vs θ_ww ({month})")
plt.show()

# =========================================
# f_LT histogram
# =========================================
plt.figure(figsize=(8,6))
plt.hist(ene_m.f_LT.values.flatten(), bins=80, density=True, alpha=0.7)
plt.xlabel(r"$f_{\mathrm{LT}}$ (fraction of dissipation)")
plt.ylabel("PDF")
plt.grid(True); plt.title(f"LT dissipation fraction f_LT ({month})")
plt.show()

# =========================================
# MAP of mean f_LT
# =========================================
f_map = ene_m.f_LT.mean("time")

plt.figure(figsize=(10,7))
plt.pcolormesh(ds.longitude, ds.latitude, f_map, shading="nearest", cmap="viridis")
plt.colorbar(label=r"$f_{\mathrm{LT}}$")
plt.title(f"Mean LT fraction (f_LT) — {month}")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.show()

# =========================================
# 2D heatmap: f_LT vs SIC & Hs
# =========================================
sic = ds_m.sic.values.flatten()
hs  = ds_m.hs.values.flatten()
flt = ene_m.f_LT.values.flatten()

plt.figure(figsize=(9,7))
plt.hist2d(sic, hs, weights=flt, bins=[40,40], cmap="magma")
plt.xlabel("Sea ice concentration")
plt.ylabel("Significant wave height Hs (m)")
plt.colorbar(label="Mean f_LT")
plt.title("f_LT vs (SIC, Hs)")
plt.show()

print("All plots generated successfully.")
#%%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# pick month & reduce to single timestep
ds1 = ds_m.isel(time=0)

proj = ccrs.NorthPolarStereo()
data_crs = ccrs.PlateCarree()

plt.figure(figsize=(8,7))
ax = plt.axes(projection=proj)
ax.set_extent([-180,180,60,90], crs=data_crs)

im = ax.pcolormesh(
    ds1.longitude,
    ds1.latitude,
    ds1.La_proj_LOW,
    transform=data_crs,
    cmap="viridis",
    shading="auto"   
)

ax.coastlines()
plt.colorbar(im, shrink=0.6)
plt.title(f"La_proj_LOW — {str(ds1.time.values)[:10]}")
plt.show()



#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Load dataset
# -----------------------------------
fn = "./langmuir_outputs_2018/Langmuir_diagnostics_plus_env_DL_2020_cor.nc"
ds = xr.open_dataset(fn)

# -----------------------------------
# Select a month (change as needed)
# -----------------------------------
target_month = "2020-03"   
ds_m = ds.sel(time=slice(f"{target_month}-01", f"{target_month}-31"))

# If dataset is daily (not hourly), use this instead:
# ds_m = ds.where(ds.time.dt.strftime("%Y-%m") == target_month, drop=True)

print("Selected month:", target_month)
print("Points in month:", ds_m.La_t.count().item())

# -----------------------------------
# Extract arrays
# -----------------------------------
La_t_vals       = ds_m.La_t.values.flatten()
La_projLOW_vals = ds_m.La_proj_LOW.values.flatten()

# Remove NaNs / infs
La_t_vals       = La_t_vals[np.isfinite(La_t_vals)]
La_projLOW_vals = La_projLOW_vals[np.isfinite(La_projLOW_vals)]

# -----------------------------------
# Plot histograms
# -----------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

bins = np.logspace(-4, 1, 60)  # covers 1e-4 to 10

ax.hist(La_t_vals, bins=bins, alpha=0.5, label=r"$La_t$", density=True)
ax.hist(La_projLOW_vals, bins=bins, alpha=0.5, label=r"$La_{\mathrm{proj,LOW}}$", density=True)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Langmuir number")
ax.set_ylabel("Probability density")
ax.set_title(f"Histogram: $La_t$ vs $La_{{proj,LOW}}$ ({target_month})")
ax.grid(True, which="both", lw=0.3)
ax.legend()

plt.tight_layout()
plt.show()

#%%  ------basic plots
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
fn = "./langmuir_outputs_2018/Langmuir_diagnostics_plus_env_DL_2018_cor.nc"
ds = xr.open_dataset(fn)

print(ds)
#%%
# pick one month for clarity

import numpy as np

# show unique year-month values
t = ds.time.dt.strftime("%Y-%m").values
print("Unique months in dataset:")
print(np.unique(t))
#%%
ds_m = ds.sel(time=slice("2020-08-01", "2020-08-31"))
print("Points in Aug 2020:", ds_m.time.size)

#%%
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1) La_t vs us0
# -------------------------------

La_t = ds_m["La_t"].values.flatten()
us0  = ds_m["us0"].values.flatten()

mask = np.isfinite(La_t) & np.isfinite(us0)
La_t = La_t[mask]
us0  = us0[mask]

plt.figure(figsize=(6,5))
plt.scatter(us0, La_t, s=1, alpha=0.25)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Surface Stokes drift $u_s$ (m/s)")
plt.ylabel("Langmuir number $La_t$")
plt.title("La_t vs us0 (Aug 2020)")
plt.grid(True)
plt.show()


# -------------------------------
# 2) La_proj_LOW vs θ_ww
# -------------------------------

theta = ds_m["theta_ww"].values.flatten()
Lap   = ds_m["La_proj_LOW"].values.flatten()

mask = np.isfinite(theta) & np.isfinite(Lap)
theta = theta[mask]
Lap = Lap[mask]

plt.figure(figsize=(6,5))
plt.scatter(theta, Lap, s=1, alpha=0.25)
plt.xlabel("Wind–wave angle $\\theta_{ww}$ (rad)")
plt.ylabel("$La_{proj,LOW}$")
plt.yscale("log")
plt.title("La_proj_LOW vs $\Theta_{ww}$ (Aug 2020)")
plt.grid(True)
plt.show()


# -------------------------------
# 3) α_LOW vs θ_ww
# -------------------------------

alpha = ds_m["alpha_LOW"].values.flatten()

mask = np.isfinite(alpha) & np.isfinite(theta)
alpha = alpha[mask]
theta2 = theta[mask]

plt.figure(figsize=(6,5))
plt.scatter(theta2, alpha, s=1, alpha=0.25)
plt.xlabel("Wind–wave angle $\\theta_{ww}$ (rad)")
plt.ylabel("Alignment angle $\\alpha_{LOW}$ (rad)")
plt.title("α_LOW vs Θ_ww (Aug 2020)")
plt.grid(True)
plt.show()


# -------------------------------
# 4) DL vs kp
# -------------------------------

DL = ds_m["DL"].values.flatten()
kp = ds_m["kp"].values.flatten()

mask = np.isfinite(DL) & np.isfinite(kp)
DL = DL[mask]
kp = kp[mask]

plt.figure(figsize=(6,5))
plt.scatter(kp, DL, s=1, alpha=0.25)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("k_p (1/m)")
plt.ylabel("Langmuir production depth $D_L$ (m)")
plt.title("DL vs kp (Aug 2020)")
plt.grid(True)
plt.show()


# -------------------------------
# 5) Histograms for sanity checks
# -------------------------------

fig,ax = plt.subplots(1,3,figsize=(15,5))

ax[0].hist(ds_m["La_t"].where(ds_m.La_t < 50).values.flatten(), bins=150, color="steelblue")
ax[0].set_title("La_t distribution (clipped at 50)")

ax[1].hist(ds_m["La_proj_LOW"].where(ds_m.La_proj_LOW < 100).values.flatten(), bins=150, color="darkorange")
ax[1].set_title("La_proj_LOW distribution (clipped at 100)")

ax[2].hist(ds_m["DL"].values.flatten(), bins=150, color="seagreen")
ax[2].set_title("DL distribution")

plt.tight_layout()
plt.show()


#%%

# ==============================================

# Constants and helper functions
# ======================================================
RHO_WATER = 1025.0
EPS = 1e-8
KAPPA = 0.4
Z1 = 1.0
G = 9.81
XI = 0.5  # Stokes penetration scale

def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def compute_alpha_low_fast(ustar, us0, theta_ww, DL, z1=Z1, kappa=KAPPA):
    eps = 1e-12
    us0_safe = xr.where(us0 > eps, us0, eps)

    # Mean Stokes shear components
    Ss = us0_safe / DL
    d_us_dx = Ss * xr.ufuncs.cos(theta_ww)
    d_us_dy = Ss * xr.ufuncs.sin(theta_ww)

    denom = (ustar / kappa) * xr.ufuncs.log(DL / z1) + d_us_dx
    num = d_us_dy
    return wrap_pi(xr.ufuncs.arctan2(num, denom))


# ======================================================
# Prepare derived wave + shear inputs
# ======================================================
print("\nComputing wave + shear inputs...")

merged_all = merged_all.assign_coords(time=merged_all.time.astype("datetime64[ns]"))

# Convert fp from log10(Hz) → Hz
fp_lin = 10 ** merged_all["fp"]

# Compute wavenumber k_p
kp = ((2 * np.pi * fp_lin) ** 2) / G

H_ML = xr.where(merged_all["MLD"] > Z1, merged_all["MLD"], Z1 + 1e-6)
DL_mld  = 0.2 * H_ML
DL_wave = XI / xr.where(kp > 0, kp, np.nan)
DL_comb = xr.where(np.isfinite(DL_wave), xr.ufuncs.minimum(DL_mld, DL_wave), DL_mld)
DL      = xr.where(DL_comb > Z1, DL_comb, Z1 + 1e-6)

# Stress → u*
tau   = np.hypot(merged_all["taux"], merged_all["tauy"])
ustar = np.sqrt(xr.where(tau < 0, 0, tau) / RHO_WATER)
us0   = np.hypot(merged_all["uuss"], merged_all["vuss"])

# Wind–wave angle
wind_angle   = np.arctan2(merged_all["vwnd"], merged_all["uwnd"])
stokes_angle = np.arctan2(merged_all["vuss"], merged_all["uuss"])
theta_ww     = wrap_pi(wind_angle - stokes_angle)

ustar_s = xr.where(ustar > EPS, ustar, EPS)
us0_s   = xr.where(us0   > EPS, us0,   EPS)

print("✔ Core variables computed")


# ======================================================
# Langmuir numbers (correct α_LOW implementation)
# ======================================================
print("\nComputing Langmuir numbers...")

alpha_LOW = compute_alpha_low_fast(ustar_s, us0_s, theta_ww, DL)

cos_ustar_proj = np.abs(np.cos(alpha_LOW))
cos_us_proj    = np.abs(np.cos(theta_ww - alpha_LOW))
cos_s          = np.abs(np.cos(theta_ww))

La_t = np.sqrt(ustar_s / us0_s)
La_proj = np.sqrt(ustar_s / (us0_s * xr.where(cos_s > 1e-6, cos_s, 1e-6)))

La_proj_LOW = np.sqrt(
    (ustar_s * cos_ustar_proj) /
    (us0_s   * xr.where(cos_us_proj > 1e-6, cos_us_proj, 1e-6))
)

print("✔ Langmuir metrics computed.")


# ======================================================
# Save Langmuir diagnostics
# ======================================================
print("\nSaving Langmuir diagnostics to NetCDF...")

# --- SAFEST WAY: write to local folder ---
output_dir = Path("./langmuir_outputs_2018")
output_dir.mkdir(parents=True, exist_ok=True)

fn_out = output_dir / "Langmuir_diagnostics_plus_env_DL_2018.nc"
print(f"Saving to {fn_out}")


keep_vars = ["sic", "sit", "hs", "dp", "MLD", "siu", "siv", "dmean"]

out_vars = xr.Dataset(
    {
        "ustar": ustar.astype("float32"),
        "us0": us0.astype("float32"),
        "theta_ww": theta_ww.astype("float32"),
        "alpha_LOW": alpha_LOW.astype("float32"),
        "La_t": La_t.astype("float32"),
        "La_proj": La_proj.astype("float32"),
        "La_proj_LOW": La_proj_LOW.astype("float32"),
        "DL": DL.astype("float32"),
        "kp": kp.astype("float32"),
        "fp": fp_lin.astype("float32"),
        **{v: merged_all[v].astype("float32") for v in keep_vars if v in merged_all},
    }
)

out_vars = out_vars.assign_coords(
    latitude=merged_all["latitude"],
    longitude=merged_all["longitude"],
    time=merged_all["time"],
).chunk({"time": 24, "y": 256, "x": 256})

encoding = {v: {"zlib": True, "complevel": 2} for v in out_vars.data_vars}

with ProgressBar():
    out_vars.to_netcdf(fn_out, engine="netcdf4", encoding=encoding)

print(f"Saved → {fn_out}")



#%%
# --- Cell: Load and filter MLD data (2018 only, aligned with common timestamps) ---

import xarray as xr
import pandas as pd

# Path to your external MLD file
MLD_FILE = "MLD_daily.nc"

# Open lazily for performance
ds_mld = xr.open_dataset(
    MLD_FILE,
    engine="netcdf4",
    chunks={"time": 1},        # one time step per chunk
    decode_times=True,         # keep actual datetimes (for filtering)
    mask_and_scale=False
)
print("Opened lazily. Variables:", list(ds_mld.data_vars))

# Check available time range
if "time" not in ds_mld:
    raise ValueError("MLD file has no 'time' coordinate.")
print("MLD time range:", str(ds_mld.time.values[0]), "→", str(ds_mld.time.values[-1]))

# Convert your existing common timestamps to datetime objects
if 'common_timestamps' not in locals():
    raise RuntimeError("Please define 'common_timestamps' (from the common 2018 step) first.")

common_datetimes = pd.to_datetime([pd.Timestamp(t) for t in common_timestamps])

#%%
#%%
import xarray as xr
import pandas as pd
from pathlib import Path
import numpy as np

# --- Path to your MLD file ---
MLD_FILE = Path("/Users/atavri/analysis/MLD_daily.nc")

# --- Open lazily for performance ---
ds_mld = xr.open_dataset(
    MLD_FILE,
    engine="netcdf4",         # good for CMEMS/GLORYS style NetCDF4
    chunks={"time": 1},        # one time step per chunk
    decode_times=False,        # we’ll decode manually below
    mask_and_scale=False
)

print(" Opened MLD dataset.")
print("Variables:", list(ds_mld.data_vars))
print("Dimensions:", ds_mld.dims)

# --- Ensure 'time' is properly decoded to datetime64 ---
if not np.issubdtype(ds_mld["time"].dtype, np.datetime64):
    if "units" in ds_mld["time"].attrs:
        ds_mld = xr.decode_cf(ds_mld)
    else:
        # Fallback: assume 3-hourly time steps starting 2018-01-01
        ds_mld = ds_mld.assign_coords(
            time=pd.date_range("2018-01-01", periods=ds_mld.sizes["time"], freq="3H")
        )

print("Time decoded:", ds_mld.time.values[0], "→", ds_mld.time.values[-1])

# --- Filter to 2018 ---
ds_mld_2018 = ds_mld.sel(time=ds_mld.time.dt.year == 2018)
print("Filtered to 2018:", ds_mld_2018.time.size, "timesteps")

# --- Convert model timestamps (already available) to datetime64 ---
common_dt = pd.to_datetime(common_timestamps)
common_days = np.unique(common_dt.normalize())  # unique days in 2018

# --- Build a boolean mask for overlapping days ---
mld_days = pd.to_datetime(ds_mld_2018.time.values).normalize()
mask = np.isin(mld_days, common_days)

# --- Filter to only those overlapping with model data ---
ds_mld_common = ds_mld_2018.isel(time=mask)
print(f"Matching MLD timesteps: {ds_mld_common.sizes['time']}")

# --- Quick sanity check ---
print(ds_mld_common)

#%%
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from dask.diagnostics import ProgressBar

fn_merged = Path("ns_ww3_MLD_merged_2018.nc")

datasets = []

print(f"Building time-resolved merged dataset with {len(common_timestamps)} timestamps...")

for key in common_timestamps:
    try:
        ts = pd.to_datetime(key, format="%Y%m%dT%H%M%S")

        ns_ds = ns_dict_common[key]
        ww3_ds = xr.Dataset(ww3_dict_common[key])
        mld_ds = ds_mld_common.sel(time=ts, method="nearest")

        ds_merged = xr.merge([ns_ds, ww3_ds, mld_ds],
                             compat="override", join="inner")
        ds_merged = ds_merged.expand_dims(time=[np.datetime64(ts)])
        datasets.append(ds_merged)

    except Exception as e:
        print(f"Skipping {key}: {e}")
        continue

print(f"Successfully merged {len(datasets)} timesteps.")

if not datasets:
    raise RuntimeError("No datasets were merged successfully!")

merged_all = xr.concat(datasets, dim="time").sortby("time")
merged_all = merged_all.chunk({"time": 24, "y": 256, "x": 256})

encoding = {v: {"zlib": True, "complevel": 2} for v in merged_all.data_vars}

print(f"Saving merged dataset → {fn_merged}")
with ProgressBar():
    merged_all.to_netcdf(fn_merged, engine="netcdf4", encoding=encoding)

print(f"Done! Merged dataset saved: {fn_merged}")
