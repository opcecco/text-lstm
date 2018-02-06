#!/usr/bin/env python3

import sys, random
import numpy as np

from keras.models import Sequential
from keras.layers import Input, LSTM, Dense


print('Parsing dataset...')

with open(sys.argv[1], 'r') as input_file:
	dataset = list(input_file.read())
	
vocab = sorted(list(set(dataset)))
index_to_char = dict(enumerate(vocab))
char_to_index = dict((index_to_char[i], i,) for i in index_to_char)

sequence_length = 50
sequences, output_characters = [], []

for i in range(0, len(dataset) - sequence_length, sequence_length):
	sequences.append(dataset[i:i + sequence_length])
	output_characters.append(dataset[i + sequence_length])
	
print('Vectorizing...')

x = np.zeros((len(sequences), sequence_length, len(vocab)), dtype = np.bool)
for s, seq in enumerate(sequences):
	for c, char in enumerate(seq):
		x[s, c, char_to_index[char]] = 1
		
y = np.zeros((len(sequences), len(vocab)), dtype = np.bool)
for o, out in enumerate(output_characters):
	y[o, char_to_index[out]] = 1
	
print('Building model...')

layers = 3
units = 256

model = Sequential()
model.add(LSTM(units, unroll = True, return_sequences = True, input_shape = (sequence_length, len(vocab))))

for i in range(layers - 2):
	model.add(LSTM(units, unroll = True, return_sequences = True))
	
model.add(LSTM(units, unroll = True))
# model.add(Dense(units, activation = 'relu'))
model.add(Dense(len(vocab), activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

print('Training...')

model.fit(x, y, epochs = 50, batch_size = 50, verbose = 1)




sample = 'ROMEO'
pattern = list(sample)

print('Generating...')

for i in range(500):
	
	x_sample = np.zeros((1, sequence_length, len(vocab)), dtype = np.bool)
	for c, char in enumerate(pattern):
		x_sample[0, c + sequence_length - len(pattern), char_to_index[char]] = 1
		
	preds = model.predict(x_sample)[0]
	preds = np.asarray(preds).astype('float64')
	preds = preds / np.sum(preds)
	
	choice = np.random.multinomial(1, preds, 1)
	index = np.argmax(choice)
	char = index_to_char[index]
	
	sample += char
	pattern.append(char)
	pattern = pattern[-40:]
		
print(sample)
