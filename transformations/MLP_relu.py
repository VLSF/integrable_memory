import jax.numpy as jnp
import equinox as eqx

from jax.nn import relu
from jax import random

class MLP(eqx.Module):
    weights: list
    biases: list

    def __init__(self, key, shapes):
        keys = random.split(key, (len(shapes)-1))
        self.weights = [random.normal(key, (s_out, s_in)) * 2 / jnp.sqrt(s_out + s_in) for s_in, s_out, key in zip(shapes[:-1], shapes[1:], keys)]
        self.biases = [jnp.zeros((s_out,)) for s_out, in zip(shapes[1:])]

    def __call__(self, u):
        v = self.weights[0] @ u + self.biases[0]
        for w, b in zip(self.weights[1:], self.biases[1:]):
            v = relu(v)
            v = w @ v + b
        return v