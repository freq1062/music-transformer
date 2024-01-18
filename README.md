A pretty small decoder-only transformer model that I wrote using pytorch for an Extended Essay research project. 

Based on [Google's Magenta](https://magenta.tensorflow.org/music-transformer). Technically this is the third version of this model, first I tried an encoder-decoder model from [this tutorial](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj3oc3O4ueDAxVdvokEHQgCC0UQwqsBegQIGxAF&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DISNdQcPhsts&usg=AOvVaw0zMv7ihV0qPGsNVgBAtjQD&opi=89978449) and then tried the decoder-only except with regular absolute attention.

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
