import jax.numpy as jnp
import equinox as eqx

from jax import random

class charge(eqx.Module):
    loc: jnp.array
    val: jnp.array

    def __init__(self, key, N, D_in, D_out):
        keys = random.split(key)
        self.loc = random.normal(keys[0], (N, D_in))
        self.val = random.normal(keys[0], (N, D_out))

    def __call__(self, u, p):
        closest_loc = jnp.argmin(jnp.linalg.norm(jnp.expand_dims(u, 0) - self.loc, axis=1, ord=p))
        return self.val[closest_loc]

class l1_charge(charge):
    def __call__(self, u):
        super().__call__(u, 1)

class l2_charge(charge):
    def __call__(self, u):
        super().__call__(u, 2)

class l_inf_charge(charge):
    def __call__(self, u):
        super().__call__(u, jnp.inf)

