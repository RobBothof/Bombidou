Implementation of Alex Graves' Handwriting synthesis using Tensorflow2 + Keras Layers

- data/prepare.py 

filters short sentences, builds stroke and ascii data into python pickle file

- data/test.py 

test the prepared data, draws 5 random handwriting examples

- model/train.py

train the model

- model/generate.py

test the model with a sentence defined in the code

- model/mdn/

keras mdn layer from : https://github.com/cpmpercussion/keras-mdn-layer

- model/lstm_attention/

custom keras layer, with a RNN sublayer, splits pen-data and window-data
feeds pen-data into the RNN and combines the result with the window-data to send to the next layer 
