import json
import os
import glob
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Real

from pastax.trajectory import Location, Trajectory
from pastax.utils import distance_on_earth, meters_to_degrees


## Functions to compute pair dispersion from drifter tracks



def get_trajectory(drifter_name: str, start_time=None, horizon_days=None) -> Trajectory:
    """
    Construction of a pastax trajectory from a json file.
    
    Args:
        drifter_name (str): path + names of drifter trajectories (json format)
        start_time (datetime64, optional): Deployment time. Defaults to None.
        horizon_days (datetime64, optional): How many days we take into account. Defaults to None.

    Returns:
        Trajectory: Pastax trajectory
    """
    #with open(f"{drifters_dir}/{drifter_name}.json") as f:
    with open(f"{drifter_name}") as f:
        traj_json = json.load(f)

        ts = []
        lat = []
        lon = []
        t0 = None
        for sample in traj_json:
            t = sample["time"]
            if start_time != None:
                if np.datetime64(t) < start_time:
                    continue
                if t0 is None:
                    t0 = np.datetime64(t)
                if horizon_days != None:
                    if np.datetime64(t) > start_time + np.timedelta64(horizon_days, "D"):
                        break
            else:
                if t0 is None:
                    t0 = np.datetime64(t) 

            ts.append(t)
            lat.append(sample["lat"])
            lon.append(sample["lon"])

        return Trajectory.from_array(
            jnp.stack((jnp.asarray(lat), jnp.asarray(lon)), axis=-1), 
            jnp.asarray(np.asarray(ts).astype("datetime64[s]").astype(int))
        )
      
def create_trajectories_and_times_list(drifter_dir: str, start_time=None, horizon_days=None):
    """
    Generate a list of trajectories and its associated times

    Args:
        drifter_dir (str): Trajectories path
        start_time (datetime64, optional): Deployment time. Defaults to None.
        horizon_days (datetime64, optional): How many days we take into account. Defaults to None.

    Returns:
        tuple: List of trajectories, list of associated times 
    """
    drifter_names = glob.glob(os.path.join(drifter_dir, '*.json'))
    
    traj_list = [] 
    for d in drifter_names:
        traj = get_trajectory(d, start_time = start_time, horizon_days = horizon_days)
        traj_list.append(traj)
        
    trajs_times = []
    for traj in traj_list:
        trajs_times.append(np.asarray(traj.times.value).astype("datetime64[s]"))
        
    return traj_list, trajs_times


        
def create_mask_trajectories(traj_list: list, trajs_times: list, ts):
    """
    Generates a 2D array, where each row is a trajectory and each column is a temporal mask to determine if the drifter transmitted its position in that interval of time

    Args:
        trajs_list (list): List of pastax trajectories
        trajs_times (list): List of the times associated with the trajectories
        ts (array): Time to interpolate the positions

    Returns:
        masks time (array): Mask to determine when the drifters transmitted their position
    """
    dt = np.timedelta64(np.diff(ts)[0]/2)
    masks_time = []
    for i in range(len(traj_list)):
        traj_times = trajs_times[i] 
        mask_traj = np.zeros(len(ts))
        for j in range(len(ts)):
            for k in range(len(traj_times)):
                if traj_times[k] >= ts[j] - np.timedelta64(15, "m") and traj_times[k] <= ts[j] + np.timedelta64(15, "m"):
                    mask_traj[j] = 1
        masks_time.append(mask_traj)
    masks_time = np.array(masks_time)

    return masks_time

def temporal_interpolation(traj_list: list, masks_time, ts, trajs_times: list):
    """
    Takes a list of pastax trajectories and interpolates them in the times ts, taking into account that not all of them transmitted at those times

    Args:
        traj_list (list): List of pastax Trajectories
        masks_time (array): ask to determine when the drifters transmitted their position
        ts (array): Times to interpolate the position
        trajs_times (list): List of the times when the drifters transmitted their position

    Returns:
        interpolated_trajs (list): List of pastax Trajectories interpolated in time
    """
    
    interpolated_trajs = []

    for i in range(len(traj_list)):
        traj = traj_list[i]
        mask = masks_time[i]

        interpolated_traj_latitude  = np.interp(ts.astype(int), trajs_times[i].astype(int), traj.latitudes.value)
        interpolated_traj_longitude = np.interp(ts.astype(int), trajs_times[i].astype(int), traj.longitudes.value)

        interpolated_traj = Trajectory.from_array(
                jnp.stack((jnp.asarray(interpolated_traj_latitude), jnp.asarray(interpolated_traj_longitude)), axis=-1), 
                jnp.asarray(np.asarray(ts).astype("datetime64[s]").astype(int)))

        interpolated_trajs.append(interpolated_traj)
        
    return interpolated_trajs

def distance_pair(traj1, traj2):
    """
    Distance between each point of two pastax Trajectories

    Args:
        traj1 (Trajectory): Trajectory 1
        traj2 (Trajectory): Trajectory 2

    Returns:
        distances (State): Separation distance (in meters)
    """
    return traj1.separation_distance(traj2).states

def K_pair(dist_pair, dt):
    """
    Pair diffusivity
    
    $$ K_{ij} = \frac{1}{2}\frac{d}{dt} (D_{ij}^2) $$

    Args:
        dist_pair (State): Separation distance between a pair of drifters (in meters)
        dt (float): Time interval

    Returns:
        K: Pair diffusivity
    """
    return 0.5 * np.gradient(dist_pair._value**2)/dt

def apply_mask_KD(dist_pair, K_pair, mask1, mask2):
    """
    Determines wheter it is valid or not to calculate the pair diffusivity (if the drifters were transmitting at that time)

    Args:
        dist_pair (State): Separation distance between a pair of drifters (in meters)
        K_pair (array): Pair diffusivity
        mask1 (array): Temporal mask for Trajectory 1
        mask2 (array): Temporal mask for Trajectory 2

    Returns:
        D, K: Pair separation (masked), Pair diffusivity (masked)
    """
    positions_to_calculate_dispersion = mask1 + mask2 == 2
    dist_masked = dist_pair._value[positions_to_calculate_dispersion]
    K_masked    = K_pair[positions_to_calculate_dispersion]
    dist_masked = dist_masked[:-1]
    K_masked    = K_masked[:-1]
    return dist_masked, K_masked

def calculate_pair_separation_pair_diffusivity_and_mask_pair(interpolated_trajs, masks_time, dt):
    """
    Calculate the pair separation D and the pair diffusivity K for all the pairs of trajectories in the list, and generate a mask for each pair.

    Args:
        interpolated_trajs (list): List of pastax Trajectories interpolated in time
        masks_time (array): Temporal masks for each Trajectory
        dt (float): time interval

    Returns:
        Ds, Ks, mass_pair: Pair separation, pair diffusivity and mask
    """
    N = len(interpolated_trajs)
    N_pairs = N*(N-1)/2
    pairs_list = []
    Ds = []
    Ks = []
    mask_pair = []

    for i in range(N):
        for j in range(N):
            if i==j:
                pass
            elif [i,j] in pairs_list or [j,i] in pairs_list:
                pass 
            else:
                pairs_list.append([i,j])
                traj_i  = interpolated_trajs[i]
                traj_j  = interpolated_trajs[j]
                dist_ij = distance_pair(traj_i, traj_j)
                K_ij    = K_pair(dist_ij, dt)
                D, K = apply_mask_KD(dist_ij, K_ij, masks_time[i,:], masks_time[j,:])
                positions_to_calculate_dispersion = masks_time[i,:] + masks_time[j,:] == 2
                Ds.append(D)
                Ks.append(K)
                mask_pair.append(positions_to_calculate_dispersion)
    return Ds, Ks, mask_pair
    
def detect_outliers(traj_list: list, trajs_times: list, show_plots=False):
    
    """
    Outlier detection based on the drifter speed.
    
    Args:
        trajs_list (list): List of pastax trajectories
        trajs_times (list): List of the times associated with the trajectories
        show_plots (bool, optional): Show trajectories and velocities. Defaults to False.
    Returns:
        new_trajs_list, new_trajs_times: Same lists as input without the outliers.
    """

    new_traj_list = []
    new_trajs_times = []

    for i in range(len(traj_list)):
        traj = traj_list[i]
        time_traj = trajs_times[i]

        steps_traj = traj.steps().states.value[1:]
        dt_traj = np.diff(time_traj)
        speed_traj = steps_traj/dt_traj.astype(int)

        th = np.mean(speed_traj) + 3*np.std(speed_traj)
        mask = speed_traj<th
        mask = np.insert(mask, True,0)
        outliers_idx = np.argwhere(1-mask)

        new_time = time_traj[mask]
        new_traj_longitude  = traj.longitudes.value[mask] 
        new_traj_latitude   = traj.latitudes.value[mask] 

        new_traj =  Trajectory.from_array(
                    jnp.stack((jnp.asarray(new_traj_latitude), jnp.asarray(new_traj_longitude)), axis=-1), 
                    jnp.asarray(np.asarray(new_time).astype("datetime64[s]").astype(int)))

        new_traj_list.append(new_traj)
        new_trajs_times.append(new_time)

        if show_plots == True:
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
            ax1.plot(traj.longitudes.value, traj.latitudes.value, 'g.-')
            ax1.plot(new_traj.longitudes.value, new_traj.latitudes.value, 'k.-')
            ax1.plot(traj.longitudes.value[outliers_idx], traj.latitudes.value[outliers_idx], 'ro')
            ax2.plot(speed_traj,'g.-')
            ax2.plot(th*np.ones_like(speed_traj), 'k--')
            ax2.plot(outliers_idx-1, speed_traj[outliers_idx-1], 'ro')
            ax2.grid()
            fig.suptitle('Trajectory ' + str(i))

    return new_traj_list, new_trajs_times
    


