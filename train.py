#!/usr/bin/env python3

import sys
import numpy as np


with open(sys.argv[1], 'r') as input_file:
	dataset = list(input_file.read())
	
vocab = sorted(list(set(dataset)))
index_to_char = dict(enumerate(vocab))
char_to_index = dict((index_to_char[i], i,) for i in index_to_char)

print(index_to_char)
print(char_to_index)
exit()

sequence_length = 5
x, y = [], []

for i in range(len(dataset) - sequence_length):
	x.append(dataset[i:i + sequence_length])
	y.append(dataset[i + 1:i + sequence_length + 1])
	# print('%s : %s' % (str(x), str(y)))
	