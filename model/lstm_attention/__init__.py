"""
LSTM Layer with Attention Mechanism
written by: Rob Bothof

Provided under MIT License
"""

### nothing fancy happening here yet, we just split the data
### into pen-data and attention window.

### i send pen-data to included LSTM sub-layer
### and concat the result with the window
### to send this on to the next layer


import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd

class LSTM_ATTENTION(tf.keras.layers.Layer):
    def __init__(self, HIDDEN_UNITS, WINDOW_UNITS, GENERATE, **kwargs):
        super(LSTM_ATTENTION, self).__init__(**kwargs)
        self.hidden_units = HIDDEN_UNITS
        self.window_units = WINDOW_UNITS
        self.generate=GENERATE
        if self.generate:
            self.lstm1 = tf.keras.layers.LSTM(self.hidden_units, return_sequences=True,stateful=True)
        else:
            self.lstm1 = tf.keras.layers.LSTM(self.hidden_units, return_sequences=True)

    def build(self, input_shape):
        if self.generate:
            self.lstm1.build((1,1,3))
        else:
            self.lstm1.build((None,None,3))
        super(LSTM_ATTENTION, self).build(input_shape)

    @property
    def trainable_weights(self):
        return self.lstm1.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.lstm1.non_trainable_weights

    def call(self, x, mask=None):
        x1,window = tf.split(x,num_or_size_splits=[3,x.shape[2]-3],axis=2)
        lstm1_out = self.lstm1(x1)

        layer_out = tf.concat([lstm1_out,window],2)
        return layer_out

    def get_config(self):
        config = {
            "HIDDEN_UNITS": self.hidden_units,
            "WINDOW_UNITS": self.window_units
        }
        base_config = super(LSTM_ATTENTION, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
