#!/usr/bin/env python3

import sys, json
import numpy as np

from keras.models import load_model


def sample(model, config, samples, prime = ' '):
	
	vocab = config['vocab']
	sequence_length = config['sequence_length']
		
	index_to_char = dict(enumerate(vocab))
	char_to_index = dict((index_to_char[i], i,) for i in index_to_char)
	
	diversity = 1.0
	sample = prime
	pattern = list(sample)
	
	for i in range(samples):
		
		x_sample = np.zeros((1, sequence_length, len(vocab)), dtype = np.bool)
		for c, char in enumerate(pattern):
			x_sample[0, c + sequence_length - len(pattern), char_to_index[char]] = 1
			
		preds = model.predict(x_sample)[0]
		preds = np.asarray(preds).astype('float64')
		
		# preds = np.exp(np.log(preds) / diversity)
		preds = preds / np.sum(preds)
		choice = np.random.multinomial(1, preds, 1)
		
		index = np.argmax(choice)
		# index = np.argmax(preds)
		char = index_to_char[index]
		
		sample += char
		pattern.append(char)
		pattern = pattern[-sequence_length:]
		
	return sample
	
	
def main():
	
	with open('config.json', 'r') as vocab_file:
		config = json.loads(vocab_file.read())
		
	model = load_model('model.h5')
	
	text = sample(model, config, int(sys.argv[2]), prime = sys.argv[1])
	print(text)
	
	
if __name__ == '__main__':
	main()
	