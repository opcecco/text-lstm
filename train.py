#!/usr/bin/env python3

import sys, os, json
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


"""
Train an LSTM model to generate text
Returns a Keras model object and a configuration dict
"""
def train(dataset, sequence_length, step, units, layers, dropout, epochs, batch_size, learning_rate, save_path):

	print('Parsing dataset')

	# Massage dataset
	dataset = dataset.lower()

	# Parse every character from the dataset
	vocab = sorted(list(set(list(dataset))))
	index_to_char = dict(enumerate(vocab))
	char_to_index = dict((index_to_char[i], i,) for i in index_to_char)

	sequences, output_characters = [], []

	# Chunk up the dataset into substrings for training
	for i in range(0, len(dataset) - sequence_length, step):
		sequences.append(dataset[i:i + sequence_length])
		output_characters.append(dataset[i + sequence_length])

	print('Samples:', len(output_characters))

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

	# Build the model according to parameters  unroll = True, implementation = 2,
	model = Sequential()
	model.add(LSTM(128, return_sequences = True, input_shape = (sequence_length, len(vocab))))
	model.add(LSTM(128, return_sequences = True))
	model.add(LSTM(256))
	model.add(Dropout(dropout))
	model.add(Dense(256, activation = 'selu'))
	model.add(Dense(128, activation = 'selu'))
	model.add(Dense(len(vocab), activation = 'softmax'))

	model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = learning_rate, decay = 0.03), metrics = ['accuracy'])

	print('Training')

	# Train the model
	config = {'vocab': vocab, 'sequence_length': sequence_length}
	with open(os.path.join(save_path, 'config.json'), 'w') as vocab_file:
		vocab_file.write(json.dumps(config))

	print(model.summary())

	model.fit(x, y, epochs = epochs, batch_size = batch_size, shuffle = True, verbose = 1,
		callbacks = [
			ModelCheckpoint(filepath = os.path.join(save_path, 'model.hdf5'), monitor = 'loss', verbose = 1, save_best_only = True),
			ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 1, verbose = 1),
		]
	)


"""
Main function
"""
def main():

	# Read a dataset from a plaintext file
	with open(sys.argv[1], 'r') as input_file:
		dataset = input_file.read()

	save_path = sys.argv[2]
	if not os.path.isdir(save_path):
		os.mkdir(save_path)

	# Params
	sequence_length = 20
	step            = 20
	units           = 128
	layers          = 3
	dropout         = 0.50
	epochs          = 128
	batch_size      = 128
	learning_rate   = 0.002

	# Train the model
	train(dataset, sequence_length, step, units, layers, dropout, epochs, batch_size, learning_rate, save_path)


if __name__ == '__main__':
	main()
