import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf

import numpy as np
import time
import mdn
import lstm_attention
import random
import pickle
import matplotlib.pyplot as plt
import time
import math

weightfile = "checkpoints/bombidou_model_375.h5"
# weightfile = "checkpoints-saved/bombidou_model_375.h5"

#model parameters
HIDDEN_UNITS = 400
OUTPUT_DIMENSION = 3
MDN_MIXTURES = 20
WINDOW_MIXTURES = 10
WINDOW_WIDTH = 2

#the generated points will be scaled by this amount: 
scale_factor = 100

### MDN tuning
pi_temperature = 0.5
sigma_temp = 0.5


def zero_start_position():
    """A zeroed out start position with pen down"""
    out = np.zeros((1, 3), dtype=np.float32)
    out[ 0, 2] = 1 # set pen down.
    return out

def generate_sketch(model, start_pos, num_points=100):
     return None

def cutoff_stroke(x):
    return np.greater(x,0.5) * 1.0

alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

### construct the model

decoder = tf.keras.Sequential()
# decoder.add(tf.keras.layers.LSTM(HIDDEN_UNITS, batch_input_shape=(1,1,OUTPUT_DIMENSION), return_sequences=True, stateful=True))
decoder.add(lstm_attention.LSTM_ATTENTION(HIDDEN_UNITS, WINDOW_MIXTURES, True, batch_input_shape=(1,1,OUTPUT_DIMENSION+len(alphabet)+1)))
decoder.add(tf.keras.layers.LSTM(HIDDEN_UNITS, return_sequences=True, stateful=True))
decoder.add(tf.keras.layers.LSTM(HIDDEN_UNITS, stateful=True))
decoder.add(mdn.MDN(OUTPUT_DIMENSION, MDN_MIXTURES))

### We are using the loss function from the MDN plugin!
decoder.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMENSION,MDN_MIXTURES), optimizer=tf.keras.optimizers.Adam())
decoder.summary()

decoder.load_weights(weightfile) # load weights independently from file


# Predict a character and plot the result.
start = time.time()

r = zero_start_position()
sketch = [r.reshape(3,)]

textsample = "The brown fox jumps over the lazy dog"
# textsample = "A long time ago in a galaxy far far away"
# textsample = "underware"

for c in range(len(textsample)):

    #average number of points per character
    ppc = 30

    for s in range(ppc):
        ##create the softwindow for this point in the sequence to generate
        windowpos   = c * ppc + s
        windowwidth = ppc * WINDOW_WIDTH
        onehot = np.zeros(len(alphabet)+1)
        
        for lc in range(len(textsample)):
            local_windowpos = (lc + 0.5) * ppc - windowpos

            if (local_windowpos > windowwidth*-0.5 and local_windowpos < windowwidth*0.5):
                char_strength = math.cos(((local_windowpos / windowwidth) + 0.5) * 2.0 * math.pi) * -0.5 + 0.5
                onehot[alphabet.find(textsample[lc])+1] = char_strength

        ## combine coords with softwindow
        q = np.concatenate((r.reshape(3),onehot.reshape(64))).reshape(1,1,67)
        params = decoder(q,training=False)

        ## sample
        r = mdn.sample_from_output(params[0].numpy(), OUTPUT_DIMENSION, MDN_MIXTURES, temp=pi_temperature, sigma_temp=sigma_temp)

        ##add to the drawing array
        sketch.append(r.reshape((3,)))

sketch = np.array(sketch)
sketch[:,0:2] *= scale_factor

# round off pen_down to 0 / 1
sketch.T[2] = cutoff_stroke(sketch.T[2])

print("generated in {eltime:.2f} seconds.".format(eltime=time.time() - start))

### create the drawing
plot_strokes=sketch.copy()
plot_strokes[:,:-1] = np.cumsum(sketch[:,:-1], axis=0)

plt.figure(figsize=(30,3))
eos_preds = np.where(plot_strokes[:,-1] == 1)
eos_preds = [0] + list(eos_preds[0]) + [-1] #add start and end indices
for m in range(len(eos_preds)-1):
    start = eos_preds[m]+1
    stop = eos_preds[m+1]
    plt.plot(plot_strokes[start:stop,0], plot_strokes[start:stop,1],'b-', linewidth=2.0)

# plt.title(ascii_data[j])
plt.gca().invert_yaxis()
plt.savefig("sketch.png")  
