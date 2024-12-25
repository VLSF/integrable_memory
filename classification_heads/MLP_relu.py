import jax.numpy as jnp
import equinox as eqx

from transformations import MLP_relu
from jax.nn import log_softmax

class MLP(MLP_relu.MLP):
    def __call__(self, u):
        v = super().__call__(u)
        return log_softmax(v)