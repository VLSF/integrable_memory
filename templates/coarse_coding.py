import jax.numpy as jnp
import equinox as eqx

class coarse_coding(eqx.Module):
    phi: eqx.Module
    psi: eqx.Module
    classification_head: eqx.Module
    quantizer: eqx.Module

    def __call__(self, v):
        u = self.phi(v)
        coarse_code = self.quantizer(u)
        memory = self.psi(coarse_code)
        log_prob = self.classification_head(memory)
        return log_prob