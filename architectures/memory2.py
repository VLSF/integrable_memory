from templates import integrable_memory
from coordinate_transformations import lifting
from quantizers import cell
from classification_heads import MLP_relu

from jax import random

class memory(integrable_memory.integrable_memory):
    
    def __init__(self, key, N_features, N_layers, N_cells, classification_shapes):
        # input should be (2*N_features,)!
        keys = random.split(key)
        self.transformation = lifting.lifting(keys[0], N_features, N_layers)
        self.quantizer = cell.cell(N_cells)
        self.classification_head = MLP_relu.MLP(keys[1], classification_shapes)