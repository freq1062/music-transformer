# Description
A pretty small decoder-only transformer model that I wrote using pytorch for an Extended Essay research project. 

Based on [Google's Magenta](https://magenta.tensorflow.org/music-transformer).

This is the fourth version of the model, here is what I have tried:
1. An encoder-decoder model from [this tutorial](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj3oc3O4ueDAxVdvokEHQgCC0UQwqsBegQIGxAF&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DISNdQcPhsts&usg=AOvVaw0zMv7ihV0qPGsNVgBAtjQD&opi=89978449) 
2. Decoder-only except with regular absolute attention.
3. Added the special skewing procedure found in [this paper](https://arxiv.org/pdf/1809.04281).
4. Current revisit, I corrected some issues with dropout, added learning rate scheduling and updated the hyperparameters now that I have access to lab machines. To sequence length went from 200 -> 1024, exactly like in the aforementioned paper.

### Architecture

Sequence length(seq_len): 1024
Embedding dimensionality(d_model): 512
Depth: 6


1. Input: Midi file converted to tokens, padded to length 1024 if necessary. Truncated if too long. This is the "seed" song that the model will continue.
2. Convert to embeddings: (seq_len, d_model) and scale by sqrt(d_model)
3. Decoder block: run depth times
>   1. Relative self attention using the efficient skewing procedure
>   2. Dropout
>   3. Normalize
>   4. Fully connected layer
>   5. Normalize
4. Apply final fully connected layer to output probabilities for each token
5. Output: Choice between top-k, top-p, top-p with a section of the seed appended to decode. See ```python showcase.ipynb``` to try each of them!

### How to run

```python showcase.ipynb``` currently contains everything required to train the model on one song, Reverie by Claude Debussy; mostly as a proof of concept. I am re-training on the full Maestro dataset and will commit the model once it's finished.

