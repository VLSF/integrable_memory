import jax.numpy as jnp
import equinox as eqx

from jax.nn import relu
from jax import random
from jax.lax import scan

class AOL(eqx.Module):
    weights: list
    biases: list

    def __init__(self, key, shapes):
        keys = random.split(key, (len(shapes)-1))
        self.weights = [random.normal(key, (s_out, s_in)) * 2 / jnp.sqrt(s_out + s_in) for s_in, s_out, key in zip(shapes[:-1], shapes[1:], keys)]
        self.biases = [jnp.zeros((s_out,)) for s_out, in zip(shapes[1:])]

    def __call__(self, u, eps=1.01):
        d = jnp.sqrt(jnp.sum(jnp.abs(self.weights[0].T @ self.weights[0]), axis=1))
        v = self.weights[0] @ (u / (d*eps)) + self.biases[0]
        for w, b in zip(self.weights[1:], self.biases[1:]):
            v = relu(v)
            d = jnp.sqrt(jnp.sum(jnp.abs(w.T @ w), axis=1))
            v = w @ (v / (d*eps)) + b
        return u + v

    def inv(self, u, N_it=10, eps=1.01):
        update_coords = lambda x, y: (u + x - self.__call__(x, eps=eps), None)
        inds = jnp.arange(N_it)
        output = scan(update_coords, self.__call__(u, eps=eps), inds)[0]
        return output