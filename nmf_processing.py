import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.decomposition import NMF
# Let N be the number of principal components kept

df_angles = pd.read_csv('angles_array.csv', index_col = 0)
# angles_array is samples x features
angles_array = df_angles.to_numpy()

df_labels = pd.read_csv('label_array.csv', index_col = 0)
label_array = df_labels.to_numpy()

angles_train, angles_test, labels_train, labels_test = sklearn.model_selection.train_test_split(angles_array, label_array, test_size = 0.3, random_state = 42)

N = 20
model = NMF(n_components=N, init='random', random_state=0)
W = model.fit_transform(angles_train)

# Extract the H matrix and take its psuedo-inverse to get our projection matrix
H = model.components_
proj_mat = np.linalg.pinv(H)
proj_train = angles_train @ proj_mat
proj_test = angles_test @ proj_mat

# Export projection matrix
np.savetxt('nmf_proj.csv', proj_mat, delimiter=',', fmt='%f')