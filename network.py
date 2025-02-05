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

        self.idx = tf.concat([tf.constant(range(3, 50)), tf.constant(range(51, 63))], axis = 0)[:, tf.newaxis]

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
        input = tf.transpose(tf.concat([input[:, 0][:, tf.newaxis], input], axis = 1))
        shape = tf.constant([63, input.shape[1]])
        out = tf.scatter_nd(self.idx, input, shape)
        out = tf.reshape(tf.transpose(out), shape = (-1, 21, 3))
        return out


    
def compute_joint_angles(coords, combinations):
    # Assume coords come in the shape [batch, num_coords, 3]
    # Assume combinations come in the shape [num_combs, 3]
    # Want to return angles of the form [batch, angles]
    # Compute vectors from coords indexed at the combinations
    
    coord_combs_0 = tf.gather(indices = combinations[:, 0], params = coords,  batch_dims = 0, axis = 1)
    coord_combs_1 = tf.gather(indices = combinations[:, 1], params = coords,  batch_dims = 0, axis = 1)
    coord_combs_2 = tf.gather(indices = combinations[:, 2], params = coords,  batch_dims = 0, axis = 1)
    
    vecA = coord_combs_0 - coord_combs_1
    vecB = coord_combs_2 - coord_combs_1

    # Compute dot and cross products across the batch, num_combs dimensions
    cos_theta = einsum(vecA, vecB, 'batch num_coords num_dim, batch num_coords num_dim -> batch num_coords')/(tf.norm(vecA, axis = 2)*tf.norm(vecB, axis = 2))
    sin_theta = tf.norm(tf.linalg.cross(vecA, vecB), axis = 2)/(tf.norm(vecA, axis = 2)*tf.norm(vecB, axis = 2))

    # Compute angles using atan2
    angles = tf.math.atan2(sin_theta, cos_theta)
    return angles

def compute_finger_lengths(coords, norm = 'euclidean'):
    # hard code an array for finger joint coordinate pairs
    finger_joints = tf.constant([[0, 1],
                                 [1, 2],
                                 [2, 3],
                                 [3, 4],
                                 [0, 5],
                                 [5, 6],
                                 [6, 7],
                                 [7, 8],
                                 [0, 9],
                                 [9, 10],
                                 [10, 11],
                                 [11, 12],
                                 [0, 13],
                                 [13, 14],
                                 [14, 15],
                                 [15, 16],
                                 [0, 17],
                                 [17, 18],
                                 [18, 19],
                                 [19, 20]])
    coord_combs_0 = tf.gather(indices = finger_joints[:, 0], params = coords,  batch_dims = 0, axis = 1)
    coord_combs_1 = tf.gather(indices = finger_joints[:, 1], params = coords,  batch_dims = 0, axis = 1)
    
    lengths = tf.norm(coord_combs_0 - coord_combs_1, ord = norm, axis = 2)
    return lengths

def compute_loss(x_batch, finger_reference, combinations, model, lam = 1):
    y_batch = model(x_batch)
    y_batch_reformat = model.reformat(y_batch)
    angles_batch = compute_joint_angles(y_batch_reformat, combinations)
    finger_lengths = compute_finger_lengths(y_batch_reformat)
    
    loss = tf.math.reduce_mean((x_batch - angles_batch) ** 2) + lam * tf.math.reduce_mean((finger_lengths - finger_reference)**2)
    return loss

def save_model(model, name):
    with open(name, "wb") as file: # file is a variable for storing the newly created file, it can be anything.
        pickle.dump(model, file) # Dump function is used to write the object into the created file in byte format.

if __name__ == "__main__":
    import argparse
    import numpy as np
    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml
    import pandas as pd
    import pickle

    from tqdm import trange
    from itertools import combinations
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
    refresh_rate = config["display"]["refresh_rate"]

    # Want to compute, from 1330 angle inputs, 21 x 3 coordinates. BUT, we are fixing the 0 point to be (0, 0, 0), the 1st landmark to be (k_1, k_1, 0), and the 17th landmark to be (k_2, k_3, 0) to fix orientation
    # Hence, we will actually need 21 x 3 - 3 - 2 - 1 = 57 outputs. Make sure to interpret the output accordingly. 
    
    num_inputs = 1330
    num_outputs = 58
    
    mlp = Reconstructor(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width)
    
    
    # Testing code - see if we can compute a forward pass and a loss
    # # import a single nmf basis vector to test on
    # df = pd.read_csv("thesis/nmf_basis.csv")
    # test = df.iloc[:, 0:2]
    # test_tf = tf.constant(test.to_numpy())
    # test_tf = tf.transpose(test_tf)
    # sample_out = mlp(test_tf)
    # test_reformat = mlp.reformat(sample_out)
    # coords = tf.reshape(tf.transpose(test_reformat), [2, 21, 3])
    # # Set up combination vector so all the angles can be computed
    # landmarks = np.arange(21)
    # combos = tf.constant(np.array(list(combinations(landmarks, 3))))
    # test_angles = compute_joint_angles(coords, combos)
    # file = open('dump.txt', 'wb')
    # pickle.dump(coords.numpy(), file)
    # file.close()
    
    # import dataset 
    # df_landmarks = pd.read_csv("thesis/landmarks_array.csv", index_col = 0)
    df_angles = pd.read_csv("thesis/angles_array.csv", index_col = 0)
    # finger_defaults = 
    
    # convert to tf tensor
    # landmarks = tf.constant(df_landmarks.to_numpy().reshape(-1, 21, 3))
    # landmark_sample = landmarks[0, :, :]
    
    # import reference lengths
    reference = np.loadtxt('reference.txt')
    finger_defaults =  tf.constant(reference, dtype = tf.float32)[tf.newaxis, :]
    
    angles = tf.constant(df_angles.to_numpy(), dtype = tf.float32)
    
    
    # initialize combinations vector
    landmark_nums = np.arange(21)
    combos = tf.constant(np.array(list(combinations(landmark_nums, 3))))
    
    step_size_vec = [0.001, 0.0005, 0.0005, 0.0001, 0.0001]
    
    for epoch in range(num_epochs):
        # shuffle landmarks since they came in class order 
        # landmarks = tf.random.shuffle(landmarks)
        angles = tf.random.shuffle(angles)
        # angles = tf.random.shuffle(angles, seed = 0x43966E87BD57227011B5B03B58785EC1)
        
        # reorganize into batches
        angles_batched = tf.reshape(angles, shape = (-1, batch_size, 1330))
        bar = trange(angles_batched.shape[0])
        optimizer = tf.keras.optimizers.Adam(learning_rate = step_size)
        for batch in bar:
            # initialize Adam optimizer
            
            x_batch = angles_batched[batch, :, :]
            with tf.GradientTape() as tape:
                loss = compute_loss(x_batch, finger_defaults, combos, mlp)
            grads = tape.gradient(loss, mlp.trainable_variables)
            optimizer.apply_gradients(zip(grads, mlp.trainable_variables))
            if batch % refresh_rate == (refresh_rate - 1):
                bar.set_description(
                    f"Step {batch}; Epoch => {epoch:0.4f}, Loss => {loss.numpy().squeeze():0.4f}, step_size => {step_size:0.4f}"
                )
                bar.refresh()
        # Basic LR scheduler
        step_size = step_size_vec[epoch]
    breakpoint()
    