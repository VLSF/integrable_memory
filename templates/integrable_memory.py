import jax.numpy as jnp
import equinox as eqx

class integrable_memory(eqx.Module):
    transformation: eqx.Module
    classification_head: eqx.Module
    quantizer: eqx.Module

    def __call__(self, v):
        u = self.transformation.inv(v)
        fixed_point = self.quantizer(u)
        memory = self.transformation(fixed_point)
        log_prob = self.classification_head(memory)
        return log_prob