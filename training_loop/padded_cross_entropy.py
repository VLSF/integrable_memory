import jax.numpy as jnp
import equinox as eqx

from jax import vmap, random
from jax.tree_util import tree_map

def compute_loss_(model, feature, target):
    log_probs = model(feature)
    loss = - log_probs[target]
    return loss

def compute_loss(model, features, targets):
    return jnp.mean(vmap(compute_loss_, in_axes=(None, 0, 0))(model, features, targets))

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

def make_step_scan(carry, n, optim, pad_features):
    model, features, targets, key, opt_state = carry
    key_, key = random.split(key)
    loss, grads = compute_loss_and_grads(model, pad_features(features[n], key), targets[n])
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, key, opt_state], loss

def compute_error(carry, n, pad_features):
    model, features, targets, key = carry
    key_, key = random.split(key)
    predicted_class = jnp.argmax(model(pad_features(features[n], key)))
    is_correct = predicted_class == targets[n]
    return [model, features, targets, key], is_correct
