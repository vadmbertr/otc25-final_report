import jax.numpy as jnp


def constant_Az(tx, ty, lat):
    theta = jnp.deg2rad(45)

    rho = 1025
    omega = 7.2921e-5

    sgnf = -jnp.sign(lat)
    theta = theta * sgnf
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    f = 2 * omega * jnp.sin(jnp.deg2rad(lat))
    beta = 1 / (rho * jnp.sqrt(2 * jnp.abs(f)))

    u_e = beta * (tx * cos_theta - ty * sin_theta)
    v_e = beta * (tx * sin_theta + ty * cos_theta)

    return u_e, v_e
