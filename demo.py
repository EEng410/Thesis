import argparse
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
import pandas as pd
import pickle
import tensorflow as tf

from tqdm import trange
from itertools import combinations
from network import Reconstructor
from network import Linear

def load_model(filepath):
    try:
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        raise
    except pickle.UnpicklingError:
        print(f"Error: The file '{filepath}' is not a valid pickle file.")
        raise
    
model = load_model("model.pkl")

nmf_basis = pd.read_csv("./thesis/nmf_basis.csv")
nmf_basis = tf.transpose(tf.constant(nmf_basis))

nmf_basis_coords = tf.transpose(tf.reshape(model.reformat(model(nmf_basis)), shape = (20, -1)))

nmf_basis_coords_df = pd.DataFrame(nmf_basis_coords)

nmf_basis_coords_df.to_csv('nmf_basis_coords.csv')
breakpoint()