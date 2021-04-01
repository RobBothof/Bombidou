""" this script tests the prepared pickle data file by rendering some plots from stokepoints with corresponding ascii line as a title """

import os,sys
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt

data_filename = "training_data.pkl"
folder = "./test"
### load prepared data
f = open(data_filename,"rb")
[stroke_data, ascii_data] = pickle.load(f)
f.close()

### remove old plots in test folder
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)

### plot 5 random entries
for i in range(5):
    j = random.randrange(len(stroke_data))
    j=i
    strokes = stroke_data[j]

    limit= 100
    scale_factor = 50

    # strokes = np.minimum(strokes, limit)
    # strokes = np.maximum(strokes, -limit)
    strokes = np.array(strokes,dtype=np.float32)
    strokes[:,0:2] /= scale_factor

    ### plot

    ### convert relative coords to absolute coords
    plot_strokes=strokes.copy()
    plot_strokes[:,:-1] = np.cumsum(strokes[:,:-1], axis=0)

    plt.figure(figsize=(20,2))
    eos_preds = np.where(plot_strokes[:,-1] == 1)
    eos_preds = [0] + list(eos_preds[0]) + [-1] #add start and end indices
    for m in range(len(eos_preds)-1):
        start = eos_preds[m]+1
        stop = eos_preds[m+1]
        plt.plot(plot_strokes[start:stop,0], plot_strokes[start:stop,1],'b-', linewidth=2.0)

    plt.title(ascii_data[j])
    plt.gca().invert_yaxis()
    plt.savefig(folder + "/" + str(j) + ".png")    
