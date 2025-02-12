import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import ast
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from itertools import combinations
import random
import tensorflow.keras.backend as K
import os

one_two_layers_file='df2.csv'
three_layers_file='df3.csv'
MC_file = 'df_MontCarlo.csv'


# +
# Function to convert string tuple to tuple type tuple
def convert_to_tuple(str_tuple):
    try:
        return tuple(ast.literal_eval(str_tuple))
    except (ValueError, SyntaxError):
        # handle malformed tuples
        return None
    
# Function to convert tuples to binary vectors
def tuple_to_vector(t, length=75):
    v = np.zeros(length, dtype=int)
    v[np.array(t)] = 1  # Set indices present in tuple t to 1
    return v

# Load data for on and two arbitrary layers
df2=pd.read_csv(one_two_layers_file,index_col=0)
df2['name'] = df2['name'].apply(convert_to_tuple)

# Load data for on and two arbitrary layers
df3=pd.read_csv(three_layers_file,index_col=0)
df3['name'] = df3['name'].apply(convert_to_tuple)

# Load data for on and two arbitrary layers
df_mc=pd.read_csv(MC_file,index_col=0)
df_mc['name'] = df_mc['name'].apply(convert_to_tuple)

# +
# Initialize a matrix with NaN values
max_j = 74  # This should be your maximum j value
matrix = np.full((75, max_j + 1), np.nan)

# Loop through i values
for i in range(75):
    
    # Set the i-th column value for cases where 'name' is (i,)
    val_i = df2.loc[df2['name'] == (i,), 'mAP']
    if not val_i.empty:
        matrix[i, i] = val_i.values[0]
    
    # Loop through j values
    for j in range(max_j + 1):
        
        # Set the j-th column value for cases where 'name' is (i, j) or (j, i)
        val_ij = df2.loc[(df2['name'] == (i, j)) | (df2['name'] == (j, i)), 'mAP']
        if not val_ij.empty:
            matrix[i, j] = val_ij.values[0]

def attach_x(x):
    size = x.shape[0]        
    x_final = np.zeros((size, 75, 76))
    for j in range(size):
        #print(j)
        x_vector = x[j].reshape(-1, 1)
        #print(x_vector)
        x_final[j] = np.hstack((x_vector, matrix))
        #print(x_final[j])
    return x_final

# Predefined vector
#vector = np.array([i for i in range(75)])
#vector = np.zeros((matrix.shape[0], 1), dtype=int)


# +

def predict(model_name,x):
    model=models.load_model(model_name)
    model.summary()
    print(x)
    print(attach_x(x))
    prediction = model.predict(attach_x(x)).flatten()  # Adjust according to your data and model
    return prediction




Acc=predict(model_name='m1.h5',x=np.full((1,75),0))
Acc
# -


