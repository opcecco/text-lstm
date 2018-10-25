#!/usr/bin/env python3

import sys, json
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import tensorflow as tf


"""
Train an LSTM model to generate text
Returns a Keras model object and a configuration dict
"""
def train(dataset, sequence_length, step, units, layers, dropout, epochs, batch_size):

	print('Parsing dataset')

	# Parse every character from the dataset
	vocab = sorted(list(set(list(dataset))))
	index_to_char = dict(enumerate(vocab))
	char_to_index = dict((index_to_char[i], i,) for i in index_to_char)

	sequences, output_characters = [], []

	# Chunk up the dataset into substrings for training
	for i in range(0, len(dataset) - sequence_length, step):
		sequences.append(dataset[i:i + sequence_length])
		output_characters.append(dataset[i + sequence_length])

	print('Vectorizing')

	# Vectorize input data (one-hot encoding)
	x = np.zeros((len(sequences), sequence_length, len(vocab)), dtype = np.bool)
	for s, seq in enumerate(sequences):
		for c, char in enumerate(seq):
			x[s, c, char_to_index[char]] = 1

	# Vectorize output data (one-hot encoding)
	y = np.zeros((len(sequences), len(vocab)), dtype = np.bool)
	for o, out in enumerate(output_characters):
		y[o, char_to_index[out]] = 1

	print('Building model')

	# Build the model according to parameters
	model = Sequential()
	model.add(LSTM(units, return_sequences = True, input_shape = (sequence_length, len(vocab))))
	model.add(Dropout(dropout))

	for i in range(layers - 1):
		model.add(LSTM(units, return_sequences = bool(i < layers - 2)))
		model.add(Dropout(dropout))

	model.add(Dense(len(vocab), activation = 'softmax'))

	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

	print('Training')

	# Train the model
	config = {'vocab': vocab, 'sequence_length': sequence_length}
	model.summary()
	model.fit(x, y, epochs = epochs, batch_size = batch_size, validation_split = 0.01, shuffle = True, verbose = 1)

	return model, config


"""
Main function
"""
def main():

	# Read a dataset from a plaintext file
	with open(sys.argv[1], 'r') as input_file:
		dataset = input_file.read()


	# Params
	sequence_length = 50
	step            = 50
	units           = 400
	layers          = 2
	dropout         = 0.20
	epochs          = 100
	batch_size      = 50

	# Train the model
	model, config = train(dataset, sequence_length, step, units, layers, dropout, epochs, batch_size)

	# Save the model and configuration settings to disk
	with open('config.json', 'w') as vocab_file:
		vocab_file.write(json.dumps(config))

	model.save('model.h5')


if __name__ == '__main__':
	main()
