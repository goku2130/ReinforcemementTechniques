import tensorflow as tf
print(tf.__version__)

import gym
import time
import numpy as np

import tensorflow.keras.optimizers as ko

np.random.seed(123)
tf.random.set_seed(123)

class DQNModel(tf.keras.Model):


    def __init__(self, num_actions, name):
        super().__init__(name = name)
        self.fc1 = tf.keras.layers.Layer(32, activation = 'relu', kernel_initializer = 'he_uniform')
        self.fc2 = tf.keras.layers.Layer(64, activation = 'relu', kernel_initializer = 'he_uniform')
        self.fc3 = tf.keras.layers.Layer(16, activation = 'relu', kernel_initializer = 'he_uniform')
        self.logits = tf.keras.layers.Dense(num_actions, name='q_values')
