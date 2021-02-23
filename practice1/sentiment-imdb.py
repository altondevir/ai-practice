#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:40:53 2021

@author: alejandro
"""

from tensorflow import keras
from tensorflow.keras import datasets, layers, models, preprocessing

max_len = 150
n_words = 8000
vector_embedding_dim = 128
batch_size = 400
epochs = 6
hidden_layers = 64

# We load in the X_train an array of list, containing the words, max 250
# In the y it is stored the positive review as 1 and negative review as 0
(X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words=n_words)

# the dataset is still in narray of lists and we gotta make it a proper 2darray 
# in numpy to use it (pad sequences also add 0 to the beginning to keep same size)
X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)


model = models.Sequential()

# We turn all the words into vectors configured depending on the words context
model.add(layers.Embedding(n_words, vector_embedding_dim, input_length=max_len))

model.add(layers.Dropout(0.2))

# We reduce the size of the data by 1D max pooling (will take max by steps)
# Reduces the time by half and helps to increase accuracy
model.add(layers.GlobalMaxPooling1D())

# We add a Hidden layer to learn when each of the output vectors from 1D are
# suggesting a certain sentiment
model.add(layers.Dense(hidden_layers, activation='relu'))

# We get lower accuracy if we do not do this dropout
model.add(layers.Dropout(0.2))

# output layer
model.add(layers.Dense(1, activation='sigmoid'))

# checking the model structure
model.summary()

# Compile
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy']) 

# Training the model
# if we pass validation data we can check how good is the model per epoch
# 6 epochs act better than 20!
model.fit(X_train, y_train, batch_size=batch_size, 
          epochs=epochs, validation_data=(X_test, y_test)) 

# Evaluate the modelwith new entry
score = model.evaluate(X_test,y_test,batch_size=batch_size)
print ("Accuracy: ", score[1])
print ("Test score: ", score[0])