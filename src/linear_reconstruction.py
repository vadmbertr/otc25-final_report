import diffrax as dfx
import jax.numpy as jnp
import numpy as np
from pastax.gridded import Gridded
from pastax.simulator import DeterministicSimulator
from pastax.trajectory import Trajectory

from forcings import get_forcings


def get_trajectory(df, traj_id, start_time, horizon_days):
    traj_df = df[df["traj_id"] == traj_id]
    traj_df = traj_df[
        (traj_df["time"] >= start_time) & (traj_df["time"] < start_time + np.timedelta64(horizon_days, "D"))
    ]

    traj = Trajectory.from_array(
        jnp.stack((jnp.asarray(traj_df.lat), jnp.asarray(traj_df.lon)), axis=-1), 
        jnp.asarray(np.asarray(traj_df.time).astype("datetime64[s]").astype(int))
    )
    
    return traj


def get_gridded(traj):
    vardyn_ds, wind_ds, wave_ds = get_forcings()
    
    min_lat, max_lat = traj.origin.latitude.value.item() - 2, traj.origin.latitude.value.item() + 2
    min_lon, max_lon = traj.origin.longitude.value.item() - 2, traj.origin.longitude.value.item() + 2
    min_time = np.datetime64(int(traj.times.value[0].item()), "s")
    max_time = np.datetime64(int(traj.times.value[-1].item()), "s") + np.timedelta64(1, "D")

    vardyn_ds = vardyn_ds.sel(
        time=slice(min_time, max_time), latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon)
    )
    wind_ds = wind_ds.sel(
        time=slice(min_time, max_time), latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon)
    )
    wave_ds = wave_ds.sel(
        time=slice(min_time, max_time), latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon)
    )

    vardyn_gridded = Gridded.from_xarray(
        vardyn_ds, 
        {"u": "ucos", "v": "vcos"}, 
        {"time": "time", "latitude": "latitude", "longitude": "longitude"}
    )
    wind_gridded = Gridded.from_xarray(
        wind_ds, 
        {"u": "eastward_wind", "v": "northward_wind", "tx": "eastward_stress", "ty": "northward_stress"}, 
        {"time": "time", "latitude": "latitude", "longitude": "longitude"}
    )
    wave_gridded = Gridded.from_xarray(
        wave_ds, 
        {"u": "VSDX", "v": "VSDY"}, 
        {"time": "time", "latitude": "latitude", "longitude": "longitude"}
    )

    return vardyn_gridded, wind_gridded, wave_gridded


def do_integration(linear_model, forcings, x0, ts):
    simulator = DeterministicSimulator()

    dt = 60  # seconds
    step_to = jnp.arange(ts[0], ts[-1] + dt, dt)
    stepsize_controller = dfx.StepTo(ts=step_to)

    reconstructed_traj = simulator(
        dynamics=linear_model, 
        args=forcings, 
        x0=x0, 
        ts=ts, 
        solver=dfx.Tsit5(), 
        stepsize_controller=stepsize_controller,
        max_steps=len(step_to)
    )

    return reconstructed_traj


def linear_reconstruction(linear_model, df, traj_id, start_time, horizon_days):
    traj = get_trajectory(df, traj_id, start_time, horizon_days)

    vardyn_gridded, wind_gridded, wave_gridded = get_gridded(traj)

    reconstructed_traj = do_integration(
        linear_model, (vardyn_gridded, wind_gridded, wave_gridded), traj.origin, traj.times.value
    )

    return traj, reconstructed_traj
