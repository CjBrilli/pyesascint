"""
geometry_utils.py

Reusable geometry helpers for DSN/VEX line-of-sight analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_horizons_los_geometry(filepath: str | Path) -> pd.DataFrame:
    """
    Load Horizons geometry needed for line-of-sight closest approach.

    Expected Horizons columns include:
    - heliocentric distance of VEX: r [AU]
    - observer-target distance: delta [AU]
    - solar elongation: S-O-T [deg]
    - heliocentric ecliptic longitude/latitude of VEX

    Returns daily geometry.
    """
    rows = []
    in_block = False

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "$$SOE" in line:
                in_block = True
                continue
            if "$$EOE" in line:
                break
            if not in_block:
                continue

            parts = line.split()
            if len(parts) < 18:
                continue

            try:
                dt = pd.to_datetime(f"{parts[0]} {parts[1]}", errors="coerce")
                if pd.isna(dt):
                    continue

                hEcl_lon_deg = float(parts[6])
                hEcl_lat_deg = float(parts[7])
                r_AU = float(parts[8])
                delta_AU = float(parts[10])

                elongation_deg = None
                for i, token in enumerate(parts):
                    if token in ("/L", "/T"):
                        elongation_deg = float(parts[i - 1])
                        break

                if elongation_deg is None:
                    continue

                rows.append({
                    "datetime": dt,
                    "hEcl_lon_deg": hEcl_lon_deg,
                    "hEcl_lat_deg": hEcl_lat_deg,
                    "r_AU": r_AU,
                    "delta_AU": delta_AU,
                    "elongation_deg": elongation_deg,
                })

            except Exception:
                continue

    geom = pd.DataFrame(rows)

    if geom.empty:
        raise ValueError(f"No usable Horizons geometry rows found in {filepath}")

    geom["day"] = geom["datetime"].dt.floor("D")

    geom_daily = (
        geom.groupby("day", as_index=False)
        .median(numeric_only=True)
        .sort_values("day")
        .reset_index(drop=True)
    )

    return geom_daily
'''
Old Geometry code
def add_los_p_point_geometry(
    windows_df: pd.DataFrame,
    geom_daily: pd.DataFrame,
    mid_col: str = "mid",
    observer_sun_distance_AU: float = 1.0,
) -> pd.DataFrame:
    """
    Add line-of-sight closest-approach geometry.

    For an Earth/Venus radio link, the closest approach of the
    signal ray path to the Sun is approximately:

        p = R_Earth-Sun * sin(SEP)

    where SEP is the Sun-Earth-spacecraft elongation angle.
    """

    out = windows_df.copy()

    if "day" not in geom_daily.columns:
        raise ValueError("geom_daily must contain a 'day' column")

    if "elongation_deg" not in geom_daily.columns:
        raise ValueError("geom_daily must contain 'elongation_deg'")

    geom = geom_daily.set_index("day").sort_index()

    t_mid = pd.to_datetime(out[mid_col]).astype("int64")
    t_geom = geom.index.astype("int64")

    out["elongation_deg"] = np.interp(
        t_mid,
        t_geom,
        geom["elongation_deg"]
    )

    sep_rad = np.deg2rad(out["elongation_deg"].astype(float))

    out["earth_sun_AU"] = observer_sun_distance_AU
    out["p_point_AU"] = observer_sun_distance_AU * np.sin(sep_rad)
    out["los_closest_from_earth_AU"] = observer_sun_distance_AU * np.cos(sep_rad)

    return out

'''

from astroquery.jplhorizons import Horizons


def _horizons_vectors(target_id, start, stop, step="10m", location="@sun"):
    """
    Query JPL Horizons Cartesian vectors in heliocentric coordinates.
    Returns dataframe with time, x_AU, y_AU, z_AU.
    """
    obj = Horizons(
        id=str(target_id),
        location=location,
        epochs={
            "start": pd.to_datetime(start).strftime("%Y-%m-%d %H:%M:%S"),
            "stop": pd.to_datetime(stop).strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
        },
    )

    vec = obj.vectors()

    time_strings = np.asarray(vec["datetime_str"], dtype=str)
    times = (
        pd.Series(time_strings)
        .str.replace("A.D. ", "", regex=False)
    )

    times = pd.to_datetime(
        times,
        format="%Y-%b-%d %H:%M:%S.%f",
        utc=True,
    )

    return pd.DataFrame({
        "time": times,
        "x_AU": np.asarray(vec["x"], dtype=float),
        "y_AU": np.asarray(vec["y"], dtype=float),
        "z_AU": np.asarray(vec["z"], dtype=float),
    })


def build_vex_earth_p_point_geometry_grid(start, stop, step="10m"):
    """
    Build true Earth–VEX line-of-sight closest-approach geometry.

    Uses heliocentric JPL Horizons vectors:
    - VEX id = -248
    - Earth id = 399

    Returns geometry dataframe with:
    time, p_point_AU, p_lon_deg, p_lat_deg, los_length_AU, projection_t_AU
    """
    vex = _horizons_vectors("-248", start, stop, step=step, location="@sun")
    earth = _horizons_vectors("399", start, stop, step=step, location="@sun")

    # Ensure same length/time grid
    n = min(len(vex), len(earth))
    vex = vex.iloc[:n].reset_index(drop=True)
    earth = earth.iloc[:n].reset_index(drop=True)

    r_vex = vex[["x_AU", "y_AU", "z_AU"]].to_numpy(float)
    r_earth = earth[["x_AU", "y_AU", "z_AU"]].to_numpy(float)

    los = r_vex - r_earth
    los_norm = np.linalg.norm(los, axis=1)
    u_los = los / los_norm[:, None]

    # Sun is origin in heliocentric coordinates
    t = -np.sum(r_earth * u_los, axis=1)
    t_clamped = np.clip(t, 0, los_norm)

    p_vec = r_earth + t_clamped[:, None] * u_los
    p_point_AU = np.linalg.norm(p_vec, axis=1)

    px, py, pz = p_vec[:, 0], p_vec[:, 1], p_vec[:, 2]

    p_lon_deg = np.degrees(np.arctan2(py, px))
    p_lon_deg = (p_lon_deg + 360) % 360

    p_lat_deg = np.degrees(np.arcsin(pz / p_point_AU))

    return pd.DataFrame({
        "time": vex["time"],
        "earth_x_AU": r_earth[:, 0],
        "earth_y_AU": r_earth[:, 1],
        "earth_z_AU": r_earth[:, 2],
        "vex_x_AU": r_vex[:, 0],
        "vex_y_AU": r_vex[:, 1],
        "vex_z_AU": r_vex[:, 2],
        "p_x_AU": p_vec[:, 0],
        "p_y_AU": p_vec[:, 1],
        "p_z_AU": p_vec[:, 2],
        "p_point_AU": p_point_AU,
        "p_lon_deg": p_lon_deg,
        "p_lat_deg": p_lat_deg,
        "los_length_AU": los_norm,
        "projection_t_AU": t,
        "projection_t_clamped_AU": t_clamped,
    })


def add_projected_p_point_geometry(windows_df, geom_df, mid_col="mid"):
    """
    Interpolate true projected P-point geometry onto window mid-times.
    """
    out = windows_df.copy()

    out[mid_col] = pd.to_datetime(out[mid_col], utc=True)
    geom = geom_df.copy()
    geom["time"] = pd.to_datetime(geom["time"], utc=True)

    x = out[mid_col].astype("int64").to_numpy()
    xp = geom["time"].astype("int64").to_numpy()

    cols = [
        "p_x_AU",
        "p_y_AU",
        "p_z_AU",
        "p_point_AU",
        "p_lon_deg",
        "p_lat_deg",
        "los_length_AU",
        "projection_t_AU",
        "projection_t_clamped_AU",
    ]

    for col in cols:
        out[col] = np.interp(x, xp, geom[col].to_numpy(float))

    return out