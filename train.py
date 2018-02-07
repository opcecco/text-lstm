#!/usr/bin/env python3

import sys, json
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop


def train(dataset, sequence_length = 50, step = 50, units = 128, layers = 2, epochs = 50, batch_size = 50, learning_rate = 0.002, implementation = 1, verbose = 0):
	
	if verbose > 0:
		print('Parsing dataset')
		
	vocab = sorted(list(set(list(dataset))))
	index_to_char = dict(enumerate(vocab))
	char_to_index = dict((index_to_char[i], i,) for i in index_to_char)
	
	sequences, output_characters = [], []
	
	for i in range(0, len(dataset) - sequence_length, step):
		sequences.append(dataset[i:i + sequence_length])
		output_characters.append(dataset[i + sequence_length])
		
	if verbose > 0:
		print('Vectorizing')
		
	x = np.zeros((len(sequences), sequence_length, len(vocab)), dtype = np.bool)
	for s, seq in enumerate(sequences):
		for c, char in enumerate(seq):
			x[s, c, char_to_index[char]] = 1
			
	y = np.zeros((len(sequences), len(vocab)), dtype = np.bool)
	for o, out in enumerate(output_characters):
		y[o, char_to_index[out]] = 1
		
	if verbose > 0:
		print('Building model')
		
	model = Sequential()
	model.add(LSTM(units, implementation = implementation, unroll = True, return_sequences = True, input_shape = (sequence_length, len(vocab))))
	
	for i in range(layers - 1):
		model.add(LSTM(units, implementation = implementation, unroll = True, return_sequences = True if i < layers - 2 else False))
		
	model.add(Dense(len(vocab), activation = 'softmax'))
	
	model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(lr = learning_rate), metrics = ['accuracy'])
	
	if verbose > 0:
		print('Training')
		
	config = {'vocab': vocab, 'sequence_length': sequence_length}
	model.fit(x, y, epochs = epochs, batch_size = batch_size, verbose = verbose)
	
	return model, config
	
	
def main():
	
	with open(sys.argv[1], 'r') as input_file:
		dataset = input_file.read()
		
	model, config = train(dataset, sequence_length = 50, step = 50, units = 128, layers = 3, epochs = 5, batch_size = 50, learning_rate = 0.002, implementation = 1, verbose = 1)
	
	with open('config.json', 'w') as vocab_file:
		vocab_file.write(json.dumps(config))
		
	model.save('model.h5')
	
	
if __name__ == '__main__':
	main()
	