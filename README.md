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

After some training on a set of IMDB movie titles and summaries, I was able to generate some interesting results.
The entire output can be found in `generated-movies.txt`. Here are just a few samples.

>Title: Outlaws of the Law
>Summary: This leads a lighthouse in the village trader and seductive Boyd Donna (Telegraph) and to pick a drink to Grand Carlo with a ship-hall performance.
>
>Title: The First Chance
>Summary: Summoned, a young man from the days have married Jo arrive at the young ward of the robberies for the mockery and tells his father that he has not lost anything to see her. He doesn't know, through all works, takes a slide, but it's part of a priest.
>
>Title: The Girl from Youth
>Summary: Businessman Lisha Thanya throws a reward for exciting on a swindle and blood and saloonkeeper. Lighting in the gang, Lindsay with the commander of an Italian pretty young couple, who is being not only a jealous for some clothes, the old English valley. Kitty now has displayed Motion Arnold's relationship with Raolf. With this football player, Rodney Steele, posing as the Inspector King of San Francisco, Margaret murders him. Complications ensue.
