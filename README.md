# Description
A pretty small decoder-only transformer model that I wrote using pytorch for an Extended Essay research project. 

Based on [Google's Magenta](https://magenta.tensorflow.org/music-transformer). The two main differences are:
1. The sequence length is 200; you can change it to generate longer sequences by just changing seq_len in config.py, but you will need to preprocess the dataset again. I don't even have a GPU so it was basically impossible for me to actually train a model bigger than that, and for the sake of research 200 was good enough for me.
2. There are no dynamics or pedal events like in the original paper; the research project I was doing was mostly about figuring out if dynamics were actually important for emotion, and I wanted to make this transformer to show that it wasn't as important as just notes. Also, again I wanted to keep it simple and reduce the complexity. Without pedal and dynamic events, a sequence length of 200 is actually more like 400 in the original representation style. 

Technically this is the third version of this model, first I tried an encoder-decoder model from [this tutorial](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj3oc3O4ueDAxVdvokEHQgCC0UQwqsBegQIGxAF&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DISNdQcPhsts&usg=AOvVaw0zMv7ihV0qPGsNVgBAtjQD&opi=89978449) and then tried the decoder-only except with regular absolute attention. 

### Architecture

1. Input: (batch_size, seq_len)
2. Embeddings: (batch_size, seq_len, d_model) and it stays this shape until the output
3. Decoder Block: run (depth) times
>  1. Relative self attention: I mostly followed [this implementation](https://jaketae.github.io/study/relative-positional-encoding/)
>  2. Dropout
>  3. Normalize
>  4. Feed forward(basically a linear layer)
>  5. Normalize again
4. Projection (batch_size seq_len, vocab_size) - it's a matrix which has a probability for each token in the vocabulary
5. Decode and show - I used the music21 library to convert the output to a score. If you don't have a score editor installed, you can change:
```python
getOutput(filename).show()
```
to:
```python
getOutput(filename).show("midi")
```
or
```python
getOutput(filename).show("text")
```
### How to run

Feel free to clone the repository and use it in another editor, I haven't looked into running it from command lines yet.

The parameters I set for this model are in config.py, and the tokens and training data from Maestro are already preprocessed in the "training_data" folder. So just unzip them in the same directory as the project clone and it should work fine. Alternatively, you can input custom training data by replacing 
```python
tokenizer, ds = createDataset([r"training_data\midiData_EVENTS.pkl", r"training_data\tokens_EVENTS.pkl"], True)
```
with 
```python
tokenizer, ds = createDataset(getFilepaths("folder with .mid files here"), False)
```

In order to start the training, just run train.py and it should start automatically.

If you finished training and would like to generate an output, scroll to the bottom of train.py and uncomment the following code:
```python
filename = "input .mid file to generate from"
getOutput(filename).show()
```
Also comment the training code which is right above that.
