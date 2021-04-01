"""
LSTM Layer with Attention Mechanism
written by: Rob Bothof

Provided under MIT License
"""

import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd

class LSTM_ATTENTION(tf.keras.layers.Layer):
    def __init__(self, HIDDEN_UNITS, WINDOW_UNITS, GENERATE, **kwargs):
        # self.hidden_units = HIDDEN_UNITS
        # with tf.name_scope('LSTM_ATTENTION'):
        #     if GENERATE:
        #         self.lstm1 = tf.keras.layers.LSTM(self.hidden_units, return_sequences=True,stateful=True)
        #     else:
        #         self.lstm1 = tf.keras.layers.LSTM(self.hidden_units, return_sequences=True)
        super(LSTM_ATTENTION, self).__init__(**kwargs)
        self.hidden_units = HIDDEN_UNITS
        self.window_units = WINDOW_UNITS
        self.generate=GENERATE
        if self.generate:
            self.lstm1 = tf.keras.layers.LSTM(self.hidden_units, return_sequences=True,stateful=True)
        else:
            self.lstm1 = tf.keras.layers.LSTM(self.hidden_units, return_sequences=True)
        
        # self.dense1 = tf.keras.layers.Dense(3 * self.window_units)


    def build(self, input_shape):
        # with tf.name_scope('lstm'):
            # self.lstm1.build(input_shape)
        if self.generate:
            self.lstm1.build((1,1,3))
            # self.dense1.build((1,1,400))
        else:
            self.lstm1.build((None,None,3))
            # self.dense1.build((None,None,400))
        super(LSTM_ATTENTION, self).build(input_shape)
        # super(LSTM_ATTENTION, self).build((None,256,38))

    @property
    def trainable_weights(self):
        # return self.lstm1.trainable_weights + self.window.trainable_weights
        return self.lstm1.trainable_weights

    @property
    def non_trainable_weights(self):
        # return self.lstm1.non_trainable_weights + self.window.non_trainable_weights
        return self.lstm1.non_trainable_weights

    def call(self, x, mask=None):
        x1,window = tf.split(x,num_or_size_splits=[3,x.shape[2]-3],axis=2)

        lstm1_out = self.lstm1(x1)

        ##get window params
        # dense1_out = self.dense1(lstm_out)
        # alpha,beta,kappa = tf.split(tf.math.softplus(dense1_out),3,axis=1)

        # # kappa = state.kappa + kappa / 25.0
        # beta = tf.clip_by_value(beta, .01, np.inf)

        # ascii_steps=64


        # kappa_flat, alpha_flat, beta_flat = kappa, alpha, beta
        # kappa, alpha, beta = tf.expand_dims(kappa, 2), tf.expand_dims(alpha, 2), tf.expand_dims(beta, 2)    


        layer_out = tf.concat([lstm1_out,window],2)
        # print (combined)
        # layer_out = tf.ones((1, 1, 100))
        return layer_out

    def get_config(self):
        config = {
            "HIDDEN_UNITS": self.hidden_units,
            "WINDOW_UNITS": self.window_units
        }
        base_config = super(LSTM_ATTENTION, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
