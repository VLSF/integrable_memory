import jax.numpy as jnp
import equinox as eqx

class cell(eqx.Module):
    N: jnp.array

    def __init__(self, N):
        self.N = jnp.array(N).astype(jnp.int32)

    def __call__(self, u):
        return (jnp.ceil(jnp.remainder(u, 1.0) * self.N) - 0.5) / self.N