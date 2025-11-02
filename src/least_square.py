import numpy as np
import pandas as pd
from scipy.optimize import nnls

from ekman_current import constant_Az
from forcings import get_forcings


def get_forcings_from_df(df):
    vardyn_ds, wind_ds, wave_ds = get_forcings()
    
    vardyn_min_time = pd.Timestamp(vardyn_ds.time.min().values, tz="UTC")
    vardyn_max_time = pd.Timestamp(vardyn_ds.time.max().values, tz="UTC")

    df = df[(df["time"] >= vardyn_min_time) & (df["time"] <= vardyn_max_time)]

    start_datetime = df["time"].min().strftime("%Y-%m-%d")
    end_datetime = df["time"].max().strftime("%Y-%m-%d")
    maximum_longitude = df["lon"].max() + 0.5
    minimum_longitude = df["lon"].min() - 0.5
    maximum_latitude = df["lat"].max() + 0.5
    minimum_latitude = df["lat"].min() - 0.5
    
    wind_ds = wind_ds.sel(
        time=slice(start_datetime, end_datetime),
        longitude=slice(minimum_longitude, maximum_longitude),
        latitude=slice(minimum_latitude, maximum_latitude),
    )

    wave_ds = wave_ds.sel(
        time=slice(start_datetime, end_datetime),
        longitude=slice(minimum_longitude, maximum_longitude),
        latitude=slice(minimum_latitude, maximum_latitude),
    )

    return df, vardyn_ds, wind_ds, wave_ds


def interp_forcings(df, vardyn_ds, wind_ds, wave_ds):
    times = pd.to_datetime(df["time"]).dt.tz_localize(None).to_numpy()
    lons = df["lon"].to_numpy()
    lats = df["lat"].to_numpy()

    # Interpolate all at once (vectorized)
    vardyn_interp = vardyn_ds[["ucos", "vcos"]].interp(
        time=("points", times),
        # time=xr.DataArray(times, dims="points"),
        longitude=("points", lons),
        latitude=("points", lats),
        method="linear",
    )

    wind_interp = wind_ds[["eastward_wind", "northward_wind", "eastward_stress", "northward_stress"]].interp(
        time=("points", times),
        longitude=("points", lons),
        latitude=("points", lats),
        method="linear",
    )

    wave_interp = wave_ds[["VSDX", "VSDY"]].interp(
        time=("points", times),
        longitude=("points", lons),
        latitude=("points", lats),
        method="linear",
    )

    df = df.copy()
    df["u_cg"] = vardyn_interp["ucos"].values
    df["v_cg"] = vardyn_interp["vcos"].values
    df["u_wind"] = wind_interp["eastward_wind"].values
    df["v_wind"] = wind_interp["northward_wind"].values
    df["tau_x"] = wind_interp["eastward_stress"].values
    df["tau_y"] = wind_interp["northward_stress"].values
    df["u_stokes"] = wave_interp["VSDX"].values
    df["v_stokes"] = wave_interp["VSDY"].values

    df = df.dropna()

    return df


def compute_ekman(df):
    u_e, v_e = constant_Az(df["tau_x"].values, df["tau_y"].values, df["lat"].values)

    df = df.copy()
    df["u_ekman"] = np.asarray(u_e)
    df["v_ekman"] = np.asarray(v_e)

    df = df.dropna()

    return df


def do_lstsq(df):
    u_obs = df["u"]
    v_obs = df["v"]

    u0 = df["u_cg"] + df["u_stokes"]
    v0 = df["v_cg"] + df["v_stokes"]

    ue = df["u_ekman"]
    ve = df["v_ekman"]

    uw = df["u_wind"].values
    vw = df["v_wind"].values
    spot = (df["type"] == "SPOT").values
    melodi = (df["type"] == "MELODI").values

    uw1 = uw * spot
    vw1 = vw * spot
    uw2 = uw * melodi
    vw2 = vw * melodi

    Ae = np.concatenate([ue.values, ve.values])[:, None]
    Aw1 = np.concatenate([uw1, vw1])[:, None]
    Aw2 = np.concatenate([uw2, vw2])[:, None]

    A = np.hstack([Ae, Aw1, Aw2])
    b = np.concatenate([u_obs - u0, v_obs - v0])

    sol = nnls(A, b)

    return sol[0]


def least_square_fit(spot_df, melodi_df):
    spot_df["type"] = "SPOT"
    melodi_df["type"] = "MELODI"
    drifters_df = pd.concat([spot_df, melodi_df], ignore_index=True)
    drifters_df = drifters_df.dropna()
   
    drifters_df, vardyn_ds, wind_ds, wave_ds = get_forcings_from_df(drifters_df)
    
    drifters_df = interp_forcings(drifters_df, vardyn_ds, wind_ds, wave_ds)
    drifters_df = compute_ekman(drifters_df)
    
    beta_hat = do_lstsq(drifters_df)

    return drifters_df, beta_hat
