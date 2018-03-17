#!/usr/bin/env python3

import sys, json
import numpy as np

from keras.models import load_model


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
		preds = np.asarray(model.predict(x_sample)[0]).astype('float64')
		preds = preds / np.sum(preds)
		choice = np.random.multinomial(1, preds, 1)
		
		index = np.argmax(choice)
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
	with open('config.json', 'r') as vocab_file:
		config = json.loads(vocab_file.read())
		
	model = load_model('model.h5')
	
	# Sample text from the model
	text = sample(model, config, int(sys.argv[2]), prime = sys.argv[1])
	print(text)
	
	
if __name__ == '__main__':
	main()
	