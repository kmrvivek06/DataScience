####################################
# Import Statements                #
####################################
import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler

####################################
# Get Data                         #
####################################
def get_data():
    # returns a T X 3 list of stock prices
    # each column is a differenr stock
    # 0 = AAPL
    # 1 = MSI
    # 2 = SBUX
    df = pd.read_csv('data/aapl_msi_sbux.csv')
    return df.values


####################################
# Replay Buffer                    #
####################################
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.uint8)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.uint8)
        self.ptr, self.size, self.max_size = 0, 0, size
        
    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        
    def sample_batch(self, batch_size=32):
        idx = np.random.randint(0, self.size, size=batch_size)
        return dict(s = self.obs1_buf[idx],
                    s2 = self.obs2_buf[idx],
                    a = self.acts_buf[idx],
                    r = self.rews_buf[idx],
                    d = self.done_buf[idx])
    
def get_scalar(env):
    # returns scikit-learn scalar object to scale the states
    # Note : you could also populate the replay buffer here
    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break
            
    scalar = StandardScaler()
    scalar.fit(states)
    return scalar
    
def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
            
def mlp(input_dim, n_action, n_hidden_layers=1, hidden_dim=32):
    # A multi-layer perceptron
    i = Input(shape=(input_dim,))
    x = i
    # hidden layers
    for _ in range(n_hidden_layers):
        x = Dense(hidden_dim, activation='relu')(x)
    # Final layer
    x = Dense(n_action)(x)
    model = Model(i,x)
    model.compile(loss='mse', optimizer='adam')
    print(model.summary())
    return model
    
    