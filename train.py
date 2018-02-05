#!/usr/bin/env python3

import sys
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense


with open(sys.argv[1], 'r') as input_file:
	dataset = list(input_file.read())
	
vocab = sorted(list(set(dataset)))
index_to_char = dict(enumerate(vocab))
char_to_index = dict((index_to_char[i], i,) for i in index_to_char)

sequence_length = 5
sequences, output_characters = [], []

for i in range(len(dataset) - sequence_length):
	sequences.append(dataset[i:i + sequence_length])
	output_characters.append(dataset[i + sequence_length])
	
x = np.zeros((len(sequences), sequence_length, len(vocab)), dtype = np.bool)
for s, seq in enumerate(sequences):
	for c, char in enumerate(s):
		x[i, c, char_to_index[char]] = 1
		
y = np.zeros((len(sequences), len(vocab)), dtype = np.bool)
for o, out in enumerate(output_characters):
	y[o, char_to_index[out]] = 1
	
model = Sequential()
model.add(LSTM(128, input_shape = (sequence_length, len(vocab))))
model.add(Dense(len(vocab)), activation = 'linear')
