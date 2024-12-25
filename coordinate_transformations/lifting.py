import jax.numpy as jnp
import equinox as eqx
import optax

from jax import jacfwd
from jax.nn import relu
from jax import random, vmap
from jax.lax import scan, dot_general

class lifting_layer(eqx.Module):
    weights: jnp.array
    bias: jnp.array

    def __init__(self, N, key):
        self.weights = random.normal(key, (2, N, N)) / jnp.sqrt(N)
        self.bias = jnp.zeros((2*N,))

    def __call__(self, x):
        N = self.weights.shape[-1]
        x = x.at[:N].set(x[:N] + self.weights[0] @ x[N:])
        x = x.at[N:].set(x[N:] + self.weights[1] @ x[:N])
        x = x + self.bias
        return x

    def inv(self, x):
        N = self.weights.shape[-1]
        x = x - self.bias
        x = x.at[N:].set(x[N:] - self.weights[1] @ x[:N])
        x = x.at[:N].set(x[:N] - self.weights[0] @ x[N:])
        return x

def leaky_relu(x):
    return x * (1.0*(x >= 0) + 0.1*(x < 0))

def inv_leaky_relu(x):
    return x * (1.0*(x >= 0) + 10.0*(x < 0))

class lifting(eqx.Module):
    lifting_layers: list

    def __init__(self, key, N, N_layers):
        self.lifting_layers = [lifting_layer(N, key) for key in random.split(key, N_layers)]

    def __call__(self, x):
        for l in self.lifting_layers:
            x = l(x)
            x = leaky_relu(x)
        return x

    def inv(self, x):
        for l in self.lifting_layers[::-1]:
            x = inv_leaky_relu(x)
            x = l.inv(x)
        return x