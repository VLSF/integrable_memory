from templates import integrable_memory
from coordinate_transformations import AOL
from quantizers import charge
from classification_heads import MLP_relu

from jax import random

class memory(integrable_memory.integrable_memory):

    def __init__(self, key, transformation_shapes, N_charges, classification_shapes, charge_order=2):
        keys = random.split(key, 3)
        self.transformation = AOL.AOL(keys[0], transformation_shapes)
        if charge_order == 1:
            self.quantizer = charge.l1_charge(keys[1], N_charges, transformation_shapes[-1])
        elif charge_order == 2:
            self.quantizer = charge.l2_charge(keys[1], N_charges, transformation_shapes[-1])
        else:
            self.quantizer = charge.l2_charge(keys[1], N_charges, transformation_shapes[-1])
        self.classification_head = MLP_relu.MLP(keys[2], classification_shapes)
