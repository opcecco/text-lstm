#!/usr/bin/env python3

import sys, os, json
import numpy as np

from keras.models import load_model


def weighted_pick(weights, temp = 1.0):

	weights = weights.astype('float64')
	preds = np.log(weights) / temp
	exp_preds = np.exp(preds)

	norm = exp_preds / np.sum(exp_preds)
	choice = np.random.multinomial(1, norm, 1)
	return np.argmax(choice)


"""
Sample a trained LSTM text model
Returns a string of generated characters
"""
def sample(model, config, samples, prime = ' '):

	# Load the configuration for text generation
	vocab = config['vocab']
	sequence_length = config['sequence_length']

	index_to_char = dict(enumerate(vocab))
	char_to_index = dict((index_to_char[i], i,) for i in index_to_char)

	sample = prime
	pattern = list(sample)

	# Generate sample, character by character
	for i in range(samples):

		# Turn the pattern of characters into a vectorized sequence of booleans
		x_sample = np.zeros((1, sequence_length, len(vocab)), dtype = np.bool)
		for c, char in enumerate(pattern):
			x_sample[0, c + sequence_length - len(pattern), char_to_index[char]] = 1

		# Use the model to generate predictions, and choose the most likely next character
		preds = model.predict(x_sample)[0]

		if pattern[-1] = ' ':
			weights = np.asarray(preds)
			index = weighted_pick(weights, 1.0)
		else:
			index = np.argmax(preds)

		char = index_to_char[index]

		# Append the character to the sample
		sample += char
		pattern.append(char)
		pattern = pattern[-sequence_length:]

	return sample


"""
Main function
"""
def main():

	# Read the configuration and model from disk
	save_path = sys.argv[1]

	with open(os.path.join(save_path, 'config.json'), 'r') as vocab_file:
		config = json.loads(vocab_file.read())

	model = load_model(os.path.join(save_path, 'model.hdf5'))

	# Sample text from the model
	text = sample(model, config, int(sys.argv[3]), prime = sys.argv[2])
	print(text)


if __name__ == '__main__':
	main()
