import copernicusmarine as cm
import xarray as xr


def get_forcings():
    vardyn_ds = xr.open_dataset("../data/VarDyn_OTC25_swot_nadirs.nc")
    vardyn_ds = vardyn_ds[["ucos", "vcos"]]

    wind_ds = cm.open_dataset(
        dataset_id="cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
        variables = ["eastward_wind", "northward_wind", "eastward_stress", "northward_stress"],
    )

    wave_ds = cm.open_dataset(
        dataset_id="cmems_mod_glo_wav_anfc_0.083deg_PT3H-i", variables=["VSDX", "VSDY"],
    )

    return vardyn_ds, wind_ds, wave_ds
