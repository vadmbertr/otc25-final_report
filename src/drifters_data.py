from functools import partial

import clouddrift as cd
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import scipy as sp
import xarray as xr


def raw_to_l0(raw_files, deployment_dates):
    def apply_per_group(df):
        df = remove_before_deployment_locations(df, deployment_dates)
        df =  compute_dt_along_trajectory(df)
        return df
    
    l0_df = pd.DataFrame()
    for f in raw_files:
        df = pd.read_json(f, convert_dates=["dateTime"])
        if len(df) == 0:
            continue
        df["id"] = df["id"].astype(str)
        df = df.rename(
            columns={"id": "drifter_id", "dateTime": "time", "latitude": "lat", "longitude": "lon"}
        )
        l0_df = pd.concat([l0_df, df], ignore_index=True)
    
    l0_df = l0_df.sort_values(by=["drifter_id", "time"]).reset_index(drop=True)
    l0_df = l0_df.groupby("drifter_id").apply(apply_per_group).reset_index(drop=True)
    l0_df = l0_df[["deploy_time", "drifter_id", "time", "lat", "lon", "dt"]]
    l0_df = l0_df.sort_values(by=["deploy_time", "drifter_id", "time"]).reset_index(drop=True)

    rowsize = l0_df["drifter_id"].value_counts(sort=False)
    l0_ds = dataframe_to_dataset(l0_df, rowsize.index.values, rowsize)

    return l0_df, l0_ds


def remove_before_deployment_locations(df, deployment_dates):
    drifter_id = df["drifter_id"].values[0]
    deploy_time = pd.to_datetime(
        deployment_dates.loc[deployment_dates["id"] == drifter_id, "deploy_time"].values[0],
        utc=True
    )

    df = df[df["time"] >= deploy_time]
    df["deploy_time"] = deploy_time

    return df


def l0_to_l1(l0_df, max_gap_hours=6, min_trajectory_days=1):
    l1_df = l0_df.groupby("drifter_id").apply(remove_spurious_gps_location).reset_index(drop=True)
    l1_df = l1_df.sort_values(by=["deploy_time", "drifter_id", "time"]).reset_index(drop=True)

    l1_ds = split_trajectories_at_gaps(l1_df, max_gap_hours)
    l1_ds = remove_small_trajectories(l1_ds, min_trajectory_days=min_trajectory_days)
    l1_ds = compute_velocity_from_position(l1_ds)

    l1_df = dataset_to_dataframe(l1_ds)
    l1_df["time"] = pd.to_datetime(l1_df["time"], utc=True)

    return l1_df, l1_ds


def remove_spurious_gps_location(df):  # following Elipot et al. 2016
    for _ in range(5):
        valid_mask = median_sd_qc_masking(df["lon"].to_numpy(), df["lat"].to_numpy())
        df = df[valid_mask].reset_index(drop=True)

    df_qc = df.copy()
    df_qc =  compute_dt_along_trajectory(df_qc)

    return df_qc


def median_sd_qc_masking(x, y):
    x = np.nan_to_num(x, np.nanmedian(x))
    y = np.nan_to_num(y, np.nanmedian(y))

    x_rolling_median = sp.ndimage.median_filter(x, size=5)
    y_rolling_median = sp.ndimage.median_filter(y, size=5)
    x_rolling_sd = sp.ndimage.generic_filter(x_rolling_median, np.std, size=5, mode="mirror")
    y_rolling_sd = sp.ndimage.generic_filter(y_rolling_median, np.std, size=5, mode="mirror")

    x_invalid_mask = (
        (x < x_rolling_median - 5 * x_rolling_sd) | 
        (x > x_rolling_median + 5 * x_rolling_sd)
    )
    y_invalid_mask = (
        (y < y_rolling_median - 5 * y_rolling_sd) | 
        (y > y_rolling_median + 5 * y_rolling_sd)
    )
    invalid_mask = x_invalid_mask | y_invalid_mask
    valid_mask = ~invalid_mask

    return valid_mask


def split_trajectories_at_gaps(df, max_gap_hours):
    rowsize = df["drifter_id"].value_counts(sort=False)
    rowsize_segmented = cd.ragged.segment(df["time"].to_numpy(), pd.Timedelta(f"{max_gap_hours}h"), rowsize.values)
    rowsize_segmented_index = np.asarray(
        [0] + [np.where(rowsize_segmented.cumsum() == e)[0].item() + 1 for e in rowsize.cumsum().values]
    )

    index = np.arange(len(df))
    nan_mask = np.isin(index, rowsize_segmented.cumsum() - 1)
    df.loc[nan_mask, "dt"] = np.nan

    drifters_id = np.empty_like(rowsize_segmented, dtype=object)
    for i, (j, k) in enumerate(zip(rowsize_segmented_index[:-1], rowsize_segmented_index[1:])):
        drifters_id[j:k] = rowsize.index.values[i]

    ds = dataframe_to_dataset(df, drifters_id, rowsize_segmented)
    return ds


def remove_small_trajectories(ds, min_trajectory_days):
    ds = cd.ragged.subset(
        ds, 
        {
            "time": lambda t: np.full_like(t, t[-1] - t[0] >= np.timedelta64(min_trajectory_days, "D"))
        },
        full_rows=True
    )
    ds["id"] = np.arange(len(ds["rowsize"]))
    return ds


def  compute_dt_along_trajectory(df):
    df = df.sort_values(by="time").reset_index(drop=True)
    dt = df["time"].diff().dt.total_seconds().shift(-1)  # time differences along a trajectory (in seconds)
    # dx, dy = earth_displacement(df[["lat", "lon"]].to_numpy())
    # u, v = dx / dt, dy / dt

    df["dt"] = dt
    # df["dx"] = dx
    # df["dy"] = dy
    # df["u"] = u
    # df["v"] = v

    return df


def earth_displacement(latlon):
    earth_radius = 6371008.8  # mean Earth radius in meters

    latlon1 = latlon[1:]
    latlon2 = latlon[:-1]

    lat1_rad = np.radians(latlon1[:, 0])
    lat2_rad = np.radians(latlon2[:, 0])
    lon1_rad = np.radians(latlon1[:, 1])
    lon2_rad = np.radians(latlon2[:, 1])

    dlat = lat1_rad - lat2_rad
    dlon = lon1_rad - lon2_rad

    dy = earth_radius * dlat
    dx = earth_radius * np.cos((lat1_rad + lat2_rad) / 2) * dlon

    dx = np.append(dx, np.nan)
    dy = np.append(dy, np.nan)

    return dx, dy


def l1_to_l2(l1_df, resample_period_hours=1):
    def do_resample(df):
        # df_num = df[["time", "lat", "lon", "u", "v"]]
        df_num = df[["time", "lat", "lon"]]
        df_num = df_num.set_index("time")
        offset_hours = df["time"].iloc[0].hour
        df_num = df_num.resample(
            "min", offset=f"{offset_hours}h"
        ).first().interpolate().resample(
            f"{resample_period_hours}h"
        ).asfreq().dropna().reset_index()
        df_num["deploy_time"] = df["deploy_time"].iloc[0]
        df_num["drifter_id"] = df["drifter_id"].iloc[0]
        df_num["traj_id"] = df["traj_id"].iloc[0]
        # df = df_num[["deploy_time", "drifter_id", "traj_id", "time", "lat", "lon", "u", "v"]]
        df = df_num[["deploy_time", "drifter_id", "traj_id", "time", "lat", "lon"]]
        return df
    
    # l2_df = l1_df.groupby("traj_id").apply(lowess_smoothing).reset_index(drop=True)
    l2_df = l1_df.groupby("traj_id").apply(do_resample).reset_index(drop=True)
    l2_df = l2_df.sort_values(by=["deploy_time", "drifter_id", "traj_id", "time"]).reset_index(drop=True)

    rowsize = l2_df["traj_id"].value_counts(sort=False)
    drifter_id = l2_df.groupby("traj_id")["drifter_id"].first().values

    l2_ds = dataframe_to_dataset(l2_df, drifter_id, rowsize)

    l2_ds = compute_velocity_from_position(l2_ds)

    l2_df = dataset_to_dataframe(l2_ds)

    return l2_df, l2_ds


def lowess_smoothing(df):  # following Elipot et al. 2016
    def lowess_1d(t, x):
        def pick_neighbors(arr, i):
            return jnp.concatenate([
                jax.lax.dynamic_slice_in_dim(arr, i - kernel_half_width, kernel_half_width),
                jax.lax.dynamic_slice_in_dim(arr, i + 1, kernel_half_width)
            ])

        def tricube_scaled(u):
            uu = jnp.minimum(1.0, jnp.abs(u))
            return (70.0 / 81.0) * (1.0 - uu**3)**3

        @partial(jax.vmap, in_axes=(0, None))
        def local_fit(i, rob_w_pad):
            t_center = t_pad[i]
            t_win = pick_neighbors(t_pad, i)
            x_win = pick_neighbors(x_pad, i)
            rob_w_win = pick_neighbors(rob_w_pad, i)

            d = jnp.abs(t_win - t_center)
            dmax = jnp.where(d.max() > 0, d.max(), 1.0)
            u = d / dmax
            kw = tricube_scaled(u) 
            w = kw * rob_w_win
            w = jnp.where(jnp.all(w <= 0), kw + eps, w)
            w_sum = w.sum()
            t_bar = jnp.sum(w * t_win) / (w_sum + eps)
            x_bar = jnp.sum(w * x_win) / (w_sum + eps)
            tt = t_win - t_bar
            xx = x_win - x_bar
            cov_tx = jnp.sum(w * tt * xx)
            var_t = jnp.sum(w * tt * tt)
            b = cov_tx / (var_t + eps)
            a = x_bar - b * t_bar
            y_pred = a + b * t_center
            return y_pred, b
        
        kernel_half_width = 2
        n_it = 3
        eps = 1e-12
        
        t_pad = jnp.pad(t, (kernel_half_width,), mode="reflect")
        x_pad = jnp.pad(x, (kernel_half_width,), mode="reflect")

        rob_w = jnp.ones_like(t)

        for it in range(n_it):
            rob_w_pad = jnp.pad(rob_w, (kernel_half_width,), mode="reflect")
            y_hat, b_hat = local_fit(jnp.arange(len(t)), rob_w_pad)

            if it == (n_it - 1):
                break

            # update robust weights
            resid = x - y_hat
            med = jnp.median(resid)
            mad = jnp.median(jnp.abs(resid - med))
            if mad < eps:
                break
            u = resid / (6.0 * mad)
            rob_w = jnp.where(jnp.abs(u) < 1.0, (1 - u**2)**2, 0.0)

        return y_hat, b_hat
    
    time = (df["time"] - pd.to_datetime("1970-01-01", utc=True)).dt.total_seconds().astype(np.int64).to_numpy()
    time -= time[0]
    ts = jnp.asarray(time)
    
    crs_4326 = CRS("WGS84")
    crs_proj = CRS("EPSG:28992")
    lonlat_to_xy = Transformer.from_crs(crs_4326, crs_proj)
    xy_to_lonlat = Transformer.from_crs(crs_proj, crs_4326)
    
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    xs, ys = lonlat_to_xy.transform(lon, lat)
    xs = jnp.asarray(xs)
    ys = jnp.asarray(ys)

    x_hat, u_hat = lowess_1d(ts, xs)
    y_hat, v_hat = lowess_1d(ts, ys)

    lon, lat = xy_to_lonlat.transform(x_hat, y_hat)

    df["lat"] = np.asarray(lat)
    df["lon"] = np.asarray(lon)
    df["u"] = np.asarray(u_hat)
    df["v"] = np.asarray(v_hat)
    
    return df


def compute_velocity_from_position(ds):
    t = (pd.to_datetime(ds.time.values, utc=True) - pd.to_datetime("1970-01-01", utc=True)).total_seconds().to_numpy()
    u, v = cd.ragged.apply_ragged(
        cd.kinematics.velocity_from_position, [ds.lon, ds.lat, t], ds.rowsize, 
        coord_system="spherical", difference_scheme="centered"
    )

    ds["u"] = ("obs", u, {
        "long_name": "Zonal velocity component",
        "units": "meters per second"
    })
    ds["v"] = ("obs", v, {
        "long_name": "Meridional velocity component",
        "units": "meters per second"
    })

    return ds


def dataframe_to_dataset(df, drifter_id, rowsize):
    ds = xr.Dataset(
        data_vars={
            "drifter_id": (
                "rows", 
                drifter_id, 
                {
                    "long_name": "Unique identifier of each drifter"
                }
            ),
            "deploy_time": (
                "rows", 
                df["deploy_time"][rowsize.cumsum() - 1],
                {
                    "long_name": "Deployment time of each drifter"
                }
            ),
            "rowsize": (
                "rows", 
                rowsize, 
                {
                    "long_name": "Number of observations per trajectory"
                }
            ),
            "lat": (
                "obs", 
                df["lat"],
                {
                    "long_name": "Latitude",
                    "units": "degrees_north"
                }
            ),
            "lon": (
                "obs", 
                df["lon"],
                {
                    "long_name": "Longitude",
                    "units": "degrees_east"
                }
            ),
        },
        coords={
            "id": (
                "rows", 
                np.arange(len(rowsize)),
                {
                    "long_name": "Index of each trajectory"
                }
            ),
            "time": (
                "obs", 
                df["time"],
                {
                    "long_name": "Time of each observation"
                }
            ),
        }
    )

    if "dt" in df:
        ds["dt"] = (
            ("obs"), 
            df["dt"],
            {
                "long_name": "Time interval between successive observations",
                "units": "seconds"
            }
        )
    if "u" in df:
        ds["u"] = (
            ("obs"), 
            df["u"],
            {
                "long_name": "Zonal velocity component",
                "units": "meters per second"
            }
        )
    if "v" in df:
        ds["v"] = (
            ("obs"), 
            df["v"],
            {
                "long_name": "Meridional velocity component",
                "units": "meters per second"
            }
        )

    return ds


def dataset_to_dataframe(ds):
    df = pd.DataFrame({
        "deploy_time": np.repeat(ds["deploy_time"].values, ds["rowsize"].values),
        "drifter_id": np.repeat(ds["drifter_id"].values, ds["rowsize"].values),
        "time": ds["time"].values,
        "lat": ds["lat"].values,
        "lon": ds["lon"].values,
    })

    if "id" in ds:
        df["traj_id"] = np.repeat(ds["id"].values, ds["rowsize"].values)
        df = df[["deploy_time", "drifter_id", "traj_id", "time", "lat", "lon"]]
    
    if "dt" in ds:
        df["dt"] = ds["dt"].values
    if "u" in ds:
        df["u"] = ds["u"].values
    if "v" in ds:
        df["v"] = ds["v"].values

    return df
