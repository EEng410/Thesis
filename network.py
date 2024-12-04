import tensorflow as tf
from einops import einsum

class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

        self.w = tf.Variable(
            rng.normal(shape=[num_inputs, num_outputs], stddev=stddev),
            trainable=True,
            name="Linear/w",
        )

        self.bias = bias

        if self.bias:
            self.b = tf.Variable(
                tf.zeros(
                    shape=[1, num_outputs],
                ),
                trainable=True,
                name="Linear/b",
            )

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return z
    
class Reconstructor(tf.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, hidden_layer_width, hidden_activation = tf.identity, output_activation = tf.identity, bias=True):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.bias = bias

        self.input_layer = Linear(num_inputs, hidden_layer_width)

        self.hidden_layers = [Linear(hidden_layer_width, hidden_layer_width) for i in range(num_hidden_layers)]

        self.output_layer = Linear(hidden_layer_width, num_outputs)

        self.idx = tf.concat([tf.constant(range(3, 50)), tf.constant(range(51, 63))])

    def __call__(self, x):
        z = tf.cast(x, 'float32')
        z = self.input_layer(z)
        
        for layer in range(self.num_hidden_layers):
            z = self.hidden_activation(self.hidden_layers[layer](z))
        p = self.output_activation(self.output_layer(z))
        return p
    
    def reformat(self, input):
        # input has shape batch, 57
        # Repeat the first index again (to force the 0-1 vector in a certain direction)
        input = tf.concat([input[:, 0], input])
        shape = tf.constant(input.shape[0], 63)
        out = tf.scatter_nd(self.idx, input, shape)
        return out


    
def compute_joint_angles(coords, combinations):
    # Assume coords come in the shape [batch, num_coords, 3]
    # Assume combinations come in the shape [num_combs, 3]
    # Want to return angles of the form [batch, angles]

    # Compute vectors from coords indexed at the combinations
    vecA = coords[:, combinations[:, 0]] - coords[:, combinations[:, 1]]
    vecB = coords[:, combinations[:, 2]] - coords[:, combinations[:, 1]]

    # Compute dot and cross products across the batch, num_combs dimensions
    cos_theta = einsum(vecA, vecB, 'batch num_coords num_dim, batch num_coords num_dim -> batch num_coords')
    sin_theta = tf.linalg.cross(vecA, vecB)

    # Compute angles using atan2
    angles = tf.math.atan2(sin_theta, cos_theta)
    return angles

if __name__ == "__main__":
    import argparse
    import numpy as np
    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml

    from tqdm import trange

    # Get some hyperparameters
    parser = argparse.ArgumentParser(
        prog="MLP",
        description="Fits a nonlinear model to some data, given a config",
    )
    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    num_epochs = config["learning"]["num_epochs"]
    step_size = config["learning"]["step_size"]
    batch_size = config["learning"]["batch_size"]
    num_hidden_layers = config["mlp"]["num_hidden_layers"]
    hidden_layer_width = config["mlp"]["hidden_layer_width"]

    # Want to compute, from 1330 angle inputs, 21 x 3 coordinates. BUT, we are fixing the 0 point to be (0, 0, 0), the 1st landmark to be (k_1, k_1, 0), and the 17th landmark to be (k_2, k_3, 0) to fix orientation
    # Hence, we will actually need 21 x 3 - 3 - 2 - 1 = 57 outputs. Make sure to interpret the output accordingly. 
    
    num_inputs = 1330
    num_outputs = 57
    
    mlp = Reconstructor(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width)