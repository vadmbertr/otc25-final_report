import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Real

from pastax.gridded import Gridded
from pastax.utils import meters_to_degrees

from ekman_current import constant_Az


class LinearCombination(eqx.Module):
    Az: Float[Array, ""] = eqx.field(converter=lambda x: jnp.asarray(x))
    beta_w: Float[Array, ""] = eqx.field(converter=lambda x: jnp.asarray(x))

    def __call__(
        self, t: Real[Array, ""], y: Float[Array, "2"], args: tuple[Gridded, Gridded, Gridded]
    ) -> Float[Array, "2"]:
        def interp(field, t, lat, lon, vars_names=["v", "u"]):
            vars_dict = field.interp(*vars_names, time=t, latitude=lat, longitude=lon)
            return jnp.asarray([vars_dict[k] for k in vars_names])
        
        def ekman():
            tau_y, tau_x = tau_yx
            u_e, v_e = constant_Az(tau_x, tau_y, latitude)
            ue_vu = jnp.asarray([u_e, v_e])
            return ue_vu / jnp.sqrt(self.Az)

        def leeway():
            u_w, v_w = uw_vu
            return jnp.asarray([u_w, v_w]) * self.beta_w

        latitude, longitude = y

        ucg_field, uw_field, uh_field = args
        
        ucg_vu = interp(ucg_field, t, latitude, longitude)
        tau_yx = interp(uw_field, t, latitude, longitude, vars_names=["ty", "tx"])
        us_vu = interp(uh_field, t, latitude, longitude)
        uw_vu = interp(uw_field, t, latitude, longitude)
        
        dlatlon = ucg_vu + ekman() + us_vu + leeway()

        if ucg_field.is_spherical_mesh and not ucg_field.use_degrees:
            dlatlon = meters_to_degrees(dlatlon, latitude=latitude)

        return dlatlon
