from templates import coarse_coding
from transformations import MLP_relu as MLP_relu_t
from quantizers import cell
from classification_heads import MLP_relu

from jax import random

class memory(coarse_coding.coarse_coding):

    def __init__(self, key, phi_shapes, psi_shapes, N_cells, classification_shapes):
        keys = random.split(key, 3)
        self.phi = MLP_relu_t.MLP(keys[0], phi_shapes)
        self.psi = MLP_relu_t.MLP(keys[1], psi_shapes)
        self.quantizer = cell.cell(N_cells)
        self.classification_head = MLP_relu.MLP(keys[2], classification_shapes)