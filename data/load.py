""" example how to load data, split in batches and prep for training """

import os,sys
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt

data_filename = "training_data.pkl"

limit= 100
scale_factor = 50
stroke_data = []
ascii_data = []
ascii_max_len = 0

batch_size = 1
batch_index = 0
tsteps = 1

U_items=65
alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

### load prepared data
f = open(data_filename,"rb")
[raw_stroke_data, raw_ascii_data] = pickle.load(f)
f.close()

## convert to numpy array, limit, scale, calc max asccii size
for i in range(len(raw_stroke_data)):
    sdata = raw_stroke_data[i]
    # sdata = np.minimum(stroke_data, limit)
    # sdata = np.maximum(stroke_data, -limit)
    sdata = np.array(sdata,dtype=np.float32)
    sdata[:,0:2] /= scale_factor

    stroke_data.append(sdata)
    ascii_data.append(raw_ascii_data[i])
    if (len(raw_ascii_data[i]) > ascii_max_len):
        ascii_max_len = len(raw_ascii_data[i])

print ()
print ("datafile loaded, length: " + str(len(stroke_data)))
print ("biggest ascii line length => " + str(ascii_max_len))
print ("batch size: " + str(batch_size) + ", " + str(int(len(stroke_data)/batch_size)) + " batches")
print ()

random_index_list = np.random.permutation(len(stroke_data))

### to sample, batch_size and tsteps should be 1
def get_batch():
    global random_index_list,batch_index,U_items,alphabet,t_steps

    x_batch = []
    y_batch = []
    ascii_list = []
    hot_list = []

    for i in range(batch_size):
        data = stroke_data[random_index_list[batch_index]]
        x_batch.append(np.copy(data[:tsteps]))
        y_batch.append(np.copy(data[1:tsteps+1]))
        ascii_list.append(ascii_data[random_index_list[batch_index]])

        batch_index +=1
        if (batch_index >= len(stroke_data)):
            random_index_list = np.random.permutation(len(stroke_data))
            batch_index=0

    for a in ascii_list:
        hot_seq = []
        for char in a:
            hot_seq.append(alphabet.find(char)+1) ### -1 not found in alphabet | +1 >> 0 is not found

        ### clamp 'seq' length to U_items if seq >= U_items
        ### pad 'seq' with zeros if seq < U_items
        if len(hot_seq) >= U_items:
            hot_seq = hot_seq[:U_items]
        else:
            hot_seq = hot_seq + [0]*(U_items - len(hot_seq))

        ###set an alphabet bit for the each char in the seq
        one_hot = np.zeros((U_items,len(alphabet)+1))
        one_hot[np.arange(U_items),hot_seq] = 1
        hot_list.append(one_hot)

    return x_batch, y_batch, ascii_list, hot_list

print(get_batch())