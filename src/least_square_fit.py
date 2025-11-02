import copernicusmarine as cm
import numpy as np
import pandas as pd
from scipy.linalg import nnls
import xarray as xr


def get_forcings(drifters_df):
    vardyn_ds = xr.open_dataset("../data/VarDyn_OTC25_swot_nadirs.nc")
    vardyn_ds = vardyn_ds[["ucos", "vcos"]]
    vardyn_min_time = pd.to_datetime(vardyn_ds.time.min().values)
    vardyn_max_time = pd.to_datetime(vardyn_ds.time.max().values)

    drifters_df = drifters_df[
        (drifters_df["time"] >= vardyn_min_time) & (drifters_df["time"] <= vardyn_max_time)
    ]

    start_datetime = drifters_df["time"].min().strftime("%Y-%m-%d")
    end_datetime = drifters_df["time"].max().strftime("%Y-%m-%d")
    maximum_longitude = drifters_df["lon"].max() + 0.25
    minimum_longitude = drifters_df["lon"].min() - 0.25
    maximum_latitude = drifters_df["lat"].max() + 0.25
    minimum_latitude = drifters_df["lat"].min() - 0.25

    wind_ds = cm.open_dataset(
        dataset_id="cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        maximum_longitude=maximum_longitude,
        minimum_longitude=minimum_longitude,
        maximum_latitude=maximum_latitude,
        minimum_latitude=minimum_latitude,
        variables = ["eastward_wind", "northward_wind"],
    )

    wave_ds = cm.open_dataset(
        dataset_id="cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        maximum_longitude=maximum_longitude,
        minimum_longitude=minimum_longitude,
        maximum_latitude=maximum_latitude,
        minimum_latitude=minimum_latitude,
        variables=["VSDX", "VSDY"],
    )

    return vardyn_ds, wind_ds, wave_ds


def interp_forcings(df, vardyn_ds, wind_ds, wave_ds):
    def interp_row(df_row):
        def do_interp(ds, var_names):
            values = []
            for var_name in var_names:
                val = ds[var_name].interp(
                    time = df_row["time"],
                    longitude = df_row["lon"],
                    latitude = df_row["lat"],
                    method = "linear",
                ).values.item()
                values.append(val)
            return values
        
        u_cg, v_cg = do_interp(vardyn_ds, ["ucos", "vcos"])
        u_wind, v_wind = do_interp(wind_ds, ["eastward_wind", "westward_wind"])
        u_stokes, v_stokes = do_interp(wave_ds, ["VSDX", "VSDY"])
        
        return u_cg, v_cg, u_wind, v_wind, u_stokes, v_stokes

    df[["u_cg", "v_cg", "u_wind", "v_wind", "u_stokes", "v_stokes"]] = df.apply(
        lambda row: interp_row(row, vardyn_ds, wind_ds, wave_ds),
        axis=1,
        result_type="expand",
    )

    return df

def compute_ekman(df):
    ekman_scaling = 1.5 / 100
    ekman_rot_angle = -np.deg2rad(45)
    rot_matrix = np.asarray([
        [np.cos(ekman_rot_angle), -np.sin(ekman_rot_angle)],
        [np.sin(ekman_rot_angle), np.cos(ekman_rot_angle)],
    ])

    uv_wind = np.asarray(df[["u_wind", "v_wind"]])
    uv_ekman = ekman_scaling * uv_wind @ rot_matrix.T
    df[["u_ekman", "v_ekman"]] = uv_ekman

    return df


def do_lstsq(df):
    u_obs = df["u"]
    v_obs = df["v"]

    u0 = df["u_cg"] + df["u_ekman"] + df["u_stokes"]
    v0 = df["v_cg"] + df["v_ekman"] + df["v_stokes"]

    uw = df["u_wind"]
    vw = df["v_wind"]

    A = np.concatenate([uw.values, vw.values])[:, None]
    b = np.concatenate([u_obs - u0, v_obs - v0])

    beta_hat, rnorm = nnls(A, b)

    return beta_hat, rnorm


def least_square_fit(drifters_df):
    drifters_df = drifters_df.dropna()

    vardyn_ds, wind_ds, wave_ds = get_forcings(drifters_df)

    drifters_df = interp_forcings(drifters_df, vardyn_ds, wind_ds, wave_ds)
    drifters_df = compute_ekman(drifters_df)

    beta_hat, rnorm = do_lstsq(drifters_df)

    return beta_hat, rnorm
