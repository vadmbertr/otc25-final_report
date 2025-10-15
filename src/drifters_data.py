import clouddrift as cd
import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr


def raw_to_l0(raw_files, deployment_dates):
    def apply_per_group(df):
        df = remove_before_deployment_locations(df, deployment_dates)
        df = compute_differences_along_trajectory(df)
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
    l0_df = l0_df[["deploy_time", "drifter_id", "time", "lat", "lon", "dt", "dx", "dy", "u", "v"]]
    l0_df = l0_df.sort_values(by=["deploy_time", "drifter_id", "time"]).reset_index(drop=True)

    rowsize = l0_df["drifter_id"].value_counts(sort=False)
    l0_ds = dataframe_to_dataset(l0_df, rowsize.index.values, rowsize)

    return l0_df, l0_ds


def l0_to_l1(l0_df, max_gap_hours=6):
    l1_df = l0_df.groupby("drifter_id").apply(remove_spurious_gps_location).reset_index(drop=True)
    l1_df = l1_df.sort_values(by=["deploy_time", "drifter_id", "time"]).reset_index(drop=True)

    rowsize = l1_df["drifter_id"].value_counts(sort=False)
    rowsize_segmented = cd.ragged.segment(l1_df["time"].to_numpy(), pd.Timedelta(f"{max_gap_hours}h"), rowsize.values)
    rowsize_segmented_index = np.asarray(
        [0] + [np.where(rowsize_segmented.cumsum() == e)[0].item() + 1 for e in rowsize.cumsum().values]
    )

    index = np.arange(len(l1_df))
    nan_mask = np.isin(index, rowsize_segmented.cumsum() - 1)
    l1_df.loc[nan_mask, ["dt", "dx", "dy", "u", "v"]] = np.nan

    drifters_id = np.empty_like(rowsize_segmented, dtype=object)
    for i, (j, k) in enumerate(zip(rowsize_segmented_index[:-1], rowsize_segmented_index[1:])):
        drifters_id[j:k] = rowsize.index.values[i]

    l1_ds = dataframe_to_dataset(l1_df, drifters_id, rowsize_segmented)
    l1_df = dataset_to_dataframe(l1_ds)
    l1_df["time"] = pd.to_datetime(l1_df["time"], utc=True)

    return l1_df, l1_ds


def remove_before_deployment_locations(df, deployment_dates):
    drifter_id = df["drifter_id"].values[0]
    deploy_time = pd.to_datetime(
        deployment_dates.loc[deployment_dates["id"] == drifter_id, "deploy_time"].values[0],
        utc=True
    )

    df = df[df["time"] >= deploy_time]
    df["deploy_time"] = deploy_time

    return df


def remove_spurious_gps_location(df):  # following Elipot et al. 2016
    for _ in range(5):
        valid_mask = median_sd_qc_masking(df["lon"].to_numpy(), df["lat"].to_numpy())
        df = df[valid_mask].reset_index(drop=True)

    df_qc = df.copy()
    df_qc = compute_differences_along_trajectory(df_qc)

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


def compute_differences_along_trajectory(df):
    df = df.sort_values(by="time").reset_index(drop=True)
    dt = df["time"].diff().dt.total_seconds().shift(-1)  # time differences along a trajectory (in seconds)
    dx, dy = earth_displacement(df[["lat", "lon"]].to_numpy())
    u, v = dx / dt, dy / dt

    df["dt"] = dt
    df["dx"] = dx
    df["dy"] = dy
    df["u"] = u
    df["v"] = v

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


def remove_small_trajectories(ds, min_trajectory_days=1):
    ds = cd.ragged.subset(
        ds, 
        {
            "time": lambda t: np.full_like(t, t[-1] - t[0] >= np.timedelta64(min_trajectory_days, "D"))
        },
        full_rows=True
    )
    ds["id"] = np.arange(len(ds["rowsize"]))
    return ds


def dataframe_to_dataset(df, drifter_id, rowsize):
    return xr.Dataset(
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
            "dt": (
                "obs", 
                df["dt"],
                {
                    "long_name": "Time difference between two consecutive observations",
                    "units": "seconds"
                }
            ),
            "dx": (
                "obs", 
                df["dx"],
                {
                    "long_name": "Zonal displacement between two consecutive observations",
                    "units": "meters"
                }
            ),
            "dy": (
                "obs", 
                df["dy"],
                {
                    "long_name": "Meridional displacement between two consecutive observations",
                    "units": "meters"
                }
            ),
            "u": (
                "obs", 
                df["u"],
                {
                    "long_name": "Zonal velocity component between two consecutive observations",
                    "units": "meters per second"
                }
            ),
            "v": (
                "obs", 
                df["v"],
                {
                    "long_name": "Meridional velocity component between two consecutive observations",
                    "units": "meters per second"
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


def dataset_to_dataframe(ds):
    df = pd.DataFrame({
        "deploy_time": np.repeat(ds["deploy_time"].values, ds["rowsize"].values),
        "drifter_id": np.repeat(ds["drifter_id"].values, ds["rowsize"].values),
        "time": ds["time"].values,
        "lat": ds["lat"].values,
        "lon": ds["lon"].values,
        "dt": ds["dt"].values,
        "dx": ds["dx"].values,
        "dy": ds["dy"].values,
        "u": ds["u"].values,
        "v": ds["v"].values,
    })

    if "id" in ds:
        df["traj_id"] = np.repeat(ds["id"].values, ds["rowsize"].values)
        df = df[["deploy_time", "drifter_id", "traj_id", "time", "lat", "lon", "dt", "dx", "dy", "u", "v"]]
    
    return df
