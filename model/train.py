import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import time
import mdn
import lstm_attention
import random
import pickle
import matplotlib.pyplot as plt
import math


alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

data_filename = "../data/training_data.pkl"
checkpoint_filepath="checkpoints/bombidou-E{epoch:02d}-L{loss:.3f}.h5"

plt.rcParams.update({'font.size': 12})
plt.style.use('dark_background')
 
#source data (x,y coords) will be divided by: 
scale_factor = 100

#length of the sequence to train on, there can be more sequences in one line of text
SEQ_LEN = 512

#number of textlines to reserve for validation
NUMBER_OF_TESTLINES = 128

#model parameters
BATCH_SIZE = 128
MAX_BATCHES = 256
WINDOW_WIDTH = 2
HIDDEN_UNITS = 400 #size of RNN layers

EPOCHS = 50 # max number of epochs
OUTPUT_DIMENSION = 3
MDN_MIXTURES = 20
WINDOW_MIXTURES = 10
DROPOUT = 0.25
LEARNING_RATE=0.0005
CLIP_VALUE=0.1

# SEED = 2345  # set random seed for reproducibility
# random.seed(SEED)
# np.random.seed(SEED)


### remove old checkpoints in test folder
for filename in os.listdir("checkpoints"):
    file_path = os.path.join("checkpoints", filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)

### GPU optimilisation
print()
for dev in tf.python.client.device_lib.list_local_devices():
    if (dev.name == tf.test.gpu_device_name()):
        print ("gpu: " + dev.physical_device_desc)
        print ("memory: " + str(dev.memory_limit))

physical_gpus = tf.config.list_physical_devices('GPU')

if physical_gpus:
    try:
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu,True)

    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

## Helper class to draw/log loss and validation
class LogLosses():
    def create(self):
        self.i = 1
        self.x = [0]
        self.losses = np.array([0])
        self.val_losses = np.array([0])
        self.fig = plt.figure(figsize=(14,16))
        self.plot()

    def update(self, loss, val_loss):
        self.losses = np.append(self.losses,min(0,loss))
        self.val_losses = np.append(self.val_losses,min(0,val_loss))
        self.x.append(self.i)       
        self.i += 1
        self.fig.clf()
        self.plot()

    def plot(self):
        plt.plot(self.x, self.losses, label="training",linewidth=2, color="red")
        plt.plot(self.x, self.val_losses, label="validation",linewidth=2,color="yellow")
        plt.legend()
        plt.grid(b=True, which='major', color='#888888', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#333333', linestyle='-')        
        plt.title("LSTM units:" + str(HIDDEN_UNITS) + "\nMDN mixtures:" + str(MDN_MIXTURES) + "\nbatch size:" + str(BATCH_SIZE) + "\nsequence length:" + str(SEQ_LEN) + "\nvalidationlines:" + str(NUMBER_OF_TESTLINES) + "\ndropout:" + str(DROPOUT) + "\n",loc='left')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.tight_layout(pad=2.0)
        plt.savefig("loss.png") 


### construct the model

inputs = tf.keras.layers.Input(shape=(SEQ_LEN,OUTPUT_DIMENSION+len(alphabet)+1), name='inputs')
# drop0_out = tf.keras.layers.Dropout(DROPOUT) (inputs)
lstm1_out = lstm_attention.LSTM_ATTENTION(HIDDEN_UNITS,WINDOW_MIXTURES,False) (inputs)
drop1_out = tf.keras.layers.Dropout(DROPOUT) (lstm1_out)
lstm2_out = tf.keras.layers.LSTM(HIDDEN_UNITS, return_sequences=True) (drop1_out)
drop2_out = tf.keras.layers.Dropout(DROPOUT) (lstm2_out)
lstm3_out = tf.keras.layers.LSTM(HIDDEN_UNITS, return_sequences=True) (drop2_out)
drop3_out = tf.keras.layers.Dropout(DROPOUT) (lstm3_out)
mdn_out = tf.keras.layers.TimeDistributed(mdn.MDN(OUTPUT_DIMENSION, MDN_MIXTURES), name='mdn')(drop3_out)

### We are using the loss function from the MDN plugin!
model = tf.keras.models.Model(inputs=inputs, outputs=mdn_out)
model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMENSION,MDN_MIXTURES), optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE,clipvalue=CLIP_VALUE))
model.summary()


### LOAD and prepare Dataset
f = open(data_filename,"rb")
[raw_stroke_data, raw_ascii_data] = pickle.load(f)
f.close()
stroke_data = []
ascii_data = []

# convert to numpy array, limit, scale, calc max asccii size
for i in range(len(raw_stroke_data)):
    sdata = raw_stroke_data[i]
    # sdata = np.minimum(stroke_data, limit)
    # sdata = np.maximum(stroke_data, -limit)
    sdata = np.array(sdata,dtype=np.float32)
    sdata[:,0:2] /= scale_factor

    stroke_data.append(sdata)
    ascii_data.append(raw_ascii_data[i])

print ()
print ("datafile loaded: " + str(len(stroke_data)) + " lines.")

### CREATE Validation SET
random_validation_list = np.random.permutation(NUMBER_OF_TESTLINES)
x_validation_set = []
y_validation_set = []

for i in range(NUMBER_OF_TESTLINES):
    stroke_seq = stroke_data[random_validation_list[i]]
    ascii_seq = ascii_data[random_validation_list[i]]

    ### precalculate the softwindow here

    ppc = len(stroke_seq)/len(ascii_seq) # average number of points for each character in this sequence
    stroke_onehot_seq = []

    for windowpos in range (len(stroke_seq)):
        windowwidth = ppc * WINDOW_WIDTH
        onehot = np.zeros(len(alphabet)+1)

        for lc in range (len(ascii_seq)):
            local_windowpos = (lc + 0.5) * ppc - windowpos

            if (local_windowpos > windowwidth*-0.5 and local_windowpos < windowwidth*0.5):
                char_strength = math.cos(((local_windowpos / windowwidth) + 0.5) * 2.0 * math.pi) * -0.5 + 0.5
                onehot[alphabet.find(ascii_seq[lc])+1] = char_strength

        # combine softwindow with pen-data
        combined = []
        combined.extend(stroke_seq[windowpos])
        combined.extend(onehot.tolist())

        stroke_onehot_seq.append(combined)   

    ### now we append the sequences to the validation array.
    ### we can get multiple sequences out of one line, and randomize them
    ### by shuffling the startpoint of each sequence

    ### create a shuffled array of startpoints for the number of sequences we can fit in this line
    randomstart = np.random.permutation(len(stroke_seq) - SEQ_LEN -2)

    # append sequence to the validation set
    for j in range(len(stroke_seq) - SEQ_LEN - 2):
        # r = randomstart[j]
        r=j
        x_validation_set.append(stroke_onehot_seq[r: r + SEQ_LEN])
        y_validation_set.append(stroke_seq[r+1: r + SEQ_LEN+1])

validation_batch_clipsize = (int(len(x_validation_set)/BATCH_SIZE)*BATCH_SIZE)
X_validation = np.array(x_validation_set[:validation_batch_clipsize])
y_validation = np.array(y_validation_set[:validation_batch_clipsize])

#free some memory
x_validation_set=None
y_validation_set=None

print ("validation_set prepared: " + str(NUMBER_OF_TESTLINES) + " lines, " + str(len(X_validation)) + " samples.")
print ()


### set up the logger
logger = LogLosses()
logger.create()


# important
#
# the complete validation set is created before actual training starts
#
# but as we have the attention window data for each pen-coord, a large dataset takes up loads of memory
# so I do not prepare the entire training set, I use incremental training and prepare a new chunk of data each iteration
# until we have gone through the entire training set. We repeat this process for the number of epochs : EPOCHS

x_train_set = []
y_train_set = []


train_start_time=time.time()

# get the size of the training set
NUMBER_OF_LINES = len(stroke_data) - NUMBER_OF_TESTLINES - 2

# create a shuffled list of indexes with the length of the training set
random_index_list = np.random.permutation(NUMBER_OF_LINES)

##start training
line=0
Iteration=0
sample=0
for ep in range(EPOCHS):
    for i in range(NUMBER_OF_LINES):
        line+=1
        stroke_seq = stroke_data[NUMBER_OF_TESTLINES + random_index_list[i]]
        ascii_seq = ascii_data[NUMBER_OF_TESTLINES + random_index_list[i]]

        ### precalculate the softwindow here

        ppc = len(stroke_seq)/len(ascii_seq) # average number of points for each character in this sequence
        stroke_onehot_seq = []

        for windowpos in range (len(stroke_seq)):
            windowwidth = ppc * WINDOW_WIDTH
            onehot = np.zeros(len(alphabet)+1)

            for lc in range (len(ascii_seq)):
                local_windowpos = (lc + 0.5) * ppc - windowpos

                if (local_windowpos > windowwidth*-0.5 and local_windowpos < windowwidth*0.5):
                    # char_strength = math.sin(((local_windowpos / windowwidth) + 0.5) * math.pi)
                    char_strength = math.cos(((local_windowpos / windowwidth) + 0.5) * 2.0 * math.pi) * -0.5 + 0.5
                    onehot[alphabet.find(ascii_seq[lc])+1] = char_strength

            # combine softwindow with pen-data
            combined = []
            combined.extend(stroke_seq[windowpos])
            combined.extend(onehot.tolist())

            stroke_onehot_seq.append(combined)   

        ### now we append the sequences to the training array.
        ### we can get multiple sequences out of one line, and randomize them
        ### by shuffling the startpoint of each sequence

        randomstart = np.random.permutation(len(stroke_seq) - SEQ_LEN -2)

        for j in range(len(stroke_seq) - SEQ_LEN - 2):
            # r = randomstart[j]
            r=j
            x_train_set.append(stroke_onehot_seq[r: r + SEQ_LEN])
            y_train_set.append(stroke_seq[r+1: r + SEQ_LEN+1])

            sample+=1

            ### actual training starts here
            if (sample >= MAX_BATCHES*BATCH_SIZE):
                if (Iteration%10==0):
                    print(" ┌───────┬────────┬───────────┬───────────┬───────────┬──────────┬──────────┐")
                    print(" │ Epoch │  Lines │ TrainTime │  Val_Time │ TotalTime │     Loss │ Val_Loss │")
                    print(" ├───────┼────────┼───────────┼───────────┼───────────┼──────────┼──────────┤")

                Iteration+=1

                X_train = np.array(x_train_set,copy=False)
                y_train = np.array(y_train_set,copy=False)

                print(" │  {:4d}".format(ep) + " │  {:5d}".format(line) + " │           │           │           │          │          │", end='\r')

                Iteration_time = time.time()
                history = model.fit(
                    X_train, 
                    y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=1, 
                    shuffle=True,
                    verbose=0,
                    callbacks=[tf.keras.callbacks.TerminateOnNaN()]
                )
                t = time.time() - Iteration_time
                sample=0

                #free some memory
                X_train = None
                y_train = None
                x_train_set = []
                y_train_set = []

                ### validate

                validation_time = time.time()
                val_loss = model.evaluate(
                    X_validation, 
                    y_validation, 
                    batch_size=BATCH_SIZE, 
                    verbose=0
                )

                ### log data
                vt = time.time() - validation_time
                tt = time.time() - train_start_time
                formatiterationtime = "{:2d}".format(int(t / 3600)) + ":{:0>2d}".format(int(t%3600 / 60)) + ":{:0>2d}".format(int(t)%60)
                formatvalidationtime = "{:2d}".format(int(vt / 3600)) + ":{:0>2d}".format(int(vt%3600 / 60)) + ":{:0>2d}".format(int(vt)%60)
                formattotaltime = "{:2d}".format(int(tt / 3600)) + ":{:0>2d}".format(int(tt%3600 / 60)) + ":{:0>2d}".format(int(tt)%60)

                print(" │  {:4d}".format(e) + " │  {:5d}".format(line) + " │  " + formatiterationtime + " │  " + formatvalidationtime + " │  " + formattotaltime + " │ {:+8.3f}".format(history.history['loss'][0]) + " │ {:+8.3f}".format(val_loss))
                logger.update(history.history['loss'][0],val_loss)

                ### save the model
                model.save("checkpoints/bombidou_model_" + str(Iteration) + ".h5") 


