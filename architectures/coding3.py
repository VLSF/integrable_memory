import equinox as eqx

from templates import coarse_coding
from transformations import MLP_relu as MLP_relu_t
from quantizers import color_charge as charge
from classification_heads import MLP_relu
from jax import random

class memory(coarse_coding.coarse_coding):

    def __init__(self, key, phi_shapes, N_charges, classification_shapes, charge_order=2):
        keys = random.split(key, 3)
        self.phi = MLP_relu_t.MLP(keys[0], phi_shapes)
        self.psi = eqx.nn.Identity()
        if charge_order == 1:
            self.quantizer = charge.l1_charge(keys[1], N_charges, phi_shapes[-1], classification_shapes[0])
        elif charge_order == 2:
            self.quantizer = charge.l2_charge(keys[1], N_charges, phi_shapes[-1], classification_shapes[0])
        else:
            self.quantizer = charge.l2_charge(keys[1], N_charges, phi_shapes[-1], classification_shapes[0])
        self.classification_head = MLP_relu.MLP(keys[2], classification_shapes)
