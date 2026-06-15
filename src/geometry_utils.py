"""
geometry_utils.py

Reusable geometry helpers for DSN/VEX line-of-sight analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd




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