# text-lstm

An LSTM network for generating text, character by character.

The model works by taking short sequences of characters as input and training to predict the next character in the sequence.
The result is a model that generates text that is "similar" to the input data.
For example, using entire Lord of the Rings trillogy as a dataset would train a model that generates text in Tolkien's fantastical style!

Try it out using some of the plaintext datasets in my [webscrapers repository](https://github.com/opcecco/webscrapers).

Requires:

```
numpy
tensorflow
keras
```

To train a new model, simply run

`python train.py <plaintext_file>`

This will save the model as two files: `model.h5` and `config.json`

Once training is finished, generate new text from your model by running

`python sample.py <starting_text> <num_characters>`
