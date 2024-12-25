from templates import integrable_memory
from coordinate_transformations import AOL
from quantizers import cell
from classification_heads import MLP_relu

from jax import random

class memory(integrable_memory.integrable_memory):

    def __init__(self, key, transformation_shapes, N_cells, classification_shapes):
        keys = random.split(key)
        self.transformation = AOL.AOL(keys[0], transformation_shapes)
        self.quantizer = cell.cell(N_cells)
        self.classification_head = MLP_relu.MLP(keys[1], classification_shapes)