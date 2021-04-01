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

# weightfile = "checkpoints/bombidou_model_520.h5"
weightfile = "checkpoints-saved/bombidou_model_520.h5"

HIDDEN_UNITS = 400
OUTPUT_DIMENSION = 3
MDN_MIXTURES = 20
WINDOW_MIXTURES = 10
WINDOW_WIDTH = 2

scale_factor = 100

# pi_temperature = 0.25
pi_temperature = 0.1
# pi_temperature = 1.0
# sigma_temp = 1.0
# sigma_temp = 0.25
sigma_temp = 0.1

def zero_start_position():
    """A zeroed out start position with pen down"""
    out = np.zeros((1, 3), dtype=np.float32)
    out[ 0, 2] = 1 # set pen down.
    return out

def generate_sketch(model, start_pos, num_points=100):
     return None

def cutoff_stroke(x):
    return np.greater(x,0.5) * 1.0

decoder = tf.keras.Sequential()
alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

# decoder.add(tf.keras.Input(shape=(1,OUTPUT_DIMENSION),batch_size=1)) 
# decoder.add(tf.keras.Input(shape=(SEQ_LEN,OUTPUT_DIMENSION))) 
# decoder.add(tf.keras.layers.Dropout(0.2,name="seq_skip1"))
# decoder.add(tf.keras.layers.LSTM(HIDDEN_UNITS, return_sequences=True, stateful=True))
# decoder.add(tf.keras.layers.Dropout(0.2,name="seq_skip2"))
# decoder.add(tf.keras.layers.LSTM(HIDDEN_UNITS, return_sequences=True, stateful=True))
# decoder.add(tf.keras.layers.Dropout(0.2,name="seq_skip3"))
# decoder.add(tf.keras.layers.LSTM(HIDDEN_UNITS, stateful=True))
# decoder.add(mdn.MDN(OUTPUT_DIMENSION, NUMBER_MIXTURES))


# decoder.add(tf.keras.layers.LSTM(HIDDEN_UNITS, batch_input_shape=(1,1,OUTPUT_DIMENSION), return_sequences=True, stateful=True))
decoder.add(lstm_attention.LSTM_ATTENTION(HIDDEN_UNITS, WINDOW_MIXTURES, True, batch_input_shape=(1,1,OUTPUT_DIMENSION+len(alphabet)+1)))
decoder.add(tf.keras.layers.LSTM(HIDDEN_UNITS, return_sequences=True, stateful=True))
decoder.add(tf.keras.layers.LSTM(HIDDEN_UNITS, stateful=True))
decoder.add(mdn.MDN(OUTPUT_DIMENSION, MDN_MIXTURES))

decoder.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMENSION,MDN_MIXTURES), optimizer=tf.keras.optimizers.Adam())
decoder.summary()

# decoder.load_weights('bombidou_model.h5') # load weights independently from file
decoder.load_weights(weightfile) # load weights independently from file


# Predict a character and plot the result.

start = time.time()

r = zero_start_position()
# print(r.reshape(1,1,3))
sketch = [r.reshape(3,)]
# z = []
# z.append(r.reshape((3,)))
# y = [z]

textsample = "The brown fox jumps over the lazy dog"
# textsample = "Thebrownfoxjumpsoverthelazydog"
# textsample = "Akiem Helmling"
# textsample = "underware"

for c in range(len(textsample)):
    # charindex = alphabet.find(textsample[c])+1
    # charindex_prev=-1
    # charindex_next=-1

    # if (c > 0):
    #     charindex_prev = alphabet.find(textsample[c-1])+1

    # if (c < len(textsample)-1):
    #     charindex_next = alphabet.find(textsample[c-1])+1

    # if (alphabet.find(textsample[c])+1 == 1):
        # ppc = 5
    # else:
    ppc = 30

    for s in range(ppc):

        windowpos   = c * ppc + s
        windowwidth = ppc * WINDOW_WIDTH

        onehot = np.zeros(len(alphabet)+1)
        
        
        for lc in range(len(textsample)):
            local_windowpos = (lc + 0.5) * ppc - windowpos

            if (local_windowpos > windowwidth*-0.5 and local_windowpos < windowwidth*0.5):
                # char_strength = math.sin(((local_windowpos / windowwidth) + 0.5) * math.pi)
                
                char_strength = math.cos(((local_windowpos / windowwidth) + 0.5) * 2.0 * math.pi) * -0.5 + 0.5
                onehot[alphabet.find(textsample[lc])+1] = char_strength

        #0 means were at the peak of the curve

        
        # onehot[alphabet.find(textsample[c])+1] = 0.8
        # if (c > 0):
        #     onehot[alphabet.find(textsample[c-1])+1] = 0.3

        # if (c > 1):
        #     onehot[alphabet.find(textsample[c-2])+1] = 0.1

        # if (c < len(textsample)-1):
        #     onehot[alphabet.find(textsample[c+1])+1] = 0.3

        # if (c < len(textsample)-2):
        #     onehot[alphabet.find(textsample[c+2])+1] = 0.1

        # print (onehot.reshape(64))
        q = np.concatenate((r.reshape(3),onehot.reshape(64))).reshape(1,1,67)
        params = decoder(q,training=False)
        # print (params[0])
        r = mdn.sample_from_output(params[0].numpy(), OUTPUT_DIMENSION, MDN_MIXTURES, temp=pi_temperature, sigma_temp=sigma_temp)
        # params = decoder.predict(r.reshape(1,1,3))
        # r = mdn.sample_from_output(params[0], OUTPUT_DIMENSION, NUMBER_MIXTURES, temp=pi_temperature, sigma_temp=sigma_temp)
        sketch.append(r.reshape((3,)))

sketch = np.array(sketch)
sketch[:,0:2] *= scale_factor

# decoder.reset_states()

# round pen_down to 0 / 1
sketch.T[2] = cutoff_stroke(sketch.T[2])


print("generated in {eltime:.2f} seconds.".format(eltime=time.time() - start))

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
