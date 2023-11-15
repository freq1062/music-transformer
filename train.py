#Train and get an output from the model. Scroll to bottom and uncomment lines of code to switch between train/output
from model import build_transformer
from dataset import causal_mask, BarDataset
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import warnings
from tqdm import tqdm
from pathlib import Path
import numpy as np
import random #If smaller dataset

#Summary stuff which I haven't written yet
import torchmetrics
from torch.utils.tensorboard import SummaryWriter

#configure.run() #Music21 visualization
import pathlib
import tensorflow as tf
import glob
import pickle
import pretty_midi
import music21

#Check if dataset is there
data_dir = pathlib.Path('data/maestro-v2.0.0')
if not data_dir.exists():
    tf.keras.utils.get_file(
    'maestro-v2.0.0-midi.zip',
    origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
    extract=True,
    cache_dir='.', cache_subdir='data',
)
filenames = glob.glob(str(data_dir/'**/*.mid*'))

DURATIONS = [1,2,4,8,16,32] #Possible durations starting with 1 = 32nd note. 32nd, 16th, 8th, quarter, half, whole
durationCodes = { #For the decoder
    1: "32nd",
    2: "16th",
    4: "eighth",
    8: "quarter",
    16: "half",
    32: "whole"
}
BEATS = list(range(0,32)) # Since smallest duration is 32nd note split each bar into 32 slots

with open(r'tokens.pkl', 'rb') as fp: tokenizer = pickle.load(fp) #load token dictionary

#Custom functions
def closest(lst, K):return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def midi_to_notes(midi_file: str):
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = []
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    for element in sorted_notes:
        start = element.start*2
        notes.append(f"{element.pitch}-{(closest(DURATIONS, (element.duration*2)*8))}-{closest(BEATS, ((start)%4)*8)}")
    return notes

def buildVocab(words,specialTokens=[]):
    words = words+specialTokens
    VOCABULARY = dict()
    counter = 0
    for ele in words:
        #print(ele)
        if ele not in list(VOCABULARY.keys()):
            VOCABULARY[str(ele)] = counter
            counter+=1
    return VOCABULARY

def createDataset(numFiles=1282): #numFiles was only from raw
    with open(r'C:\Users\alexm\Downloads\Sound Testing 3\midiData.pkl', 'rb') as fp: ds = random.choices(pickle.load(fp),k=3)
    
    #Uncomment to get notes from raw midi files 

    #ds = []
    #seq_len = get_config()["seq_len"]-3
    #notes_raw = []
    #for f in tqdm(filenames[:numFiles],desc=f"Loading {numFiles}/1282 files from maestro dataset..."): #Later: Get data from premade .pkl file
        #Combine all of the notes from files into 1 array ["note1", "note2", "note3"...]
    #    notes_raw = notes_raw + midi_to_notes(f)
    #tokenizer = buildVocab(notes_raw, ["SOS", "EOS", "PAD"])
    #notes_raw = np.split(notes_raw, np.arange(seq_len, len(notes_raw), seq_len)) #Split notes into arrays of seq_len [[seq_len], [seq_len], ...]
    #ds_len = len(notes_raw)-1
    #for i in tqdm(range(0, ds_len),desc="Splitting data into pairs"):ds.append([notes_raw[i], notes_raw[i+1]])
        #Split notes into pairs for training [ [[seq_len],[seq_len]], [[seq_len],[seq_len]], ... ]
    #return tokenizer, ds

    return ds

def saveData(): #From the raw dataset creates the pkl files which load faster
    #uncomment "from raw fles" section and call
    tokenizer, ds = createDataset()
    with open(r'C:\Users\alexm\Downloads\Sound Testing 3\tokens.pkl', 'wb') as fp:
        pickle.dump(tokenizer, fp)
        print("Saved tokens to file")
    with open(r"C:\Users\alexm\Downloads\Sound Testing 3\midiData.pkl", 'wb') as fp:
        pickle.dump(ds, fp)
        print("Saved training data to file")

def arrangeFile(filename): #Arrange an input midi file to the same format as the one the model recognizes
    ds = []
    seq_len = get_config()["seq_len"]
    notes_raw = midi_to_notes(filename)
    notes_raw = list(map(lambda x: tokenizer[x], notes_raw)) #Tokenize everything
    notes_raw = np.split(notes_raw, np.arange(seq_len-2, len(notes_raw), seq_len-2)) #[[seq_len], [seq_len], ...]

    #Create special tokens for padding
    sos_token = torch.tensor([tokenizer["EOS"]], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer["SOS"]], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer["PAD"]], dtype=torch.int64)

    for seq in notes_raw:
        enc_num_padding_tokens = seq_len - len(seq) - 2
        encoder_input = torch.cat([
            sos_token,
            torch.tensor(seq, dtype=torch.int64),
            eos_token,
            torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64),])
        encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int()
        ds.append({"encoder_input": encoder_input, "encoder_mask":encoder_mask})
    return ds

def greedy_decode(model, source, source_mask, max_len, device): # With the model and an input("source") get the output from the model
    sos_idx = tokenizer['SOS']
    eos_idx = tokenizer['EOS']

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, max_len, device, print_msg, num_examples=2):
    model.eval()
    count = 0

    source_texts = [] #Inputs to be given to the model
    expected = [] #What the actual answer was
    predicted = [] #What was predicted by the model

    #Size of the control window
    console_width = 80

    with torch.no_grad(): #Do not update weights during the evaluation
        for batch in validation_ds:
            count +=1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            print("Decoding output")
            model_out = greedy_decode(model, encoder_input, encoder_mask, max_len, device)
            print("Finished")
            func = np.vectorize(lambda x: list(tokenizer.keys())[x])
            
            source_text = " ".join( list(map(lambda x: x[0], func(torch.stack(batch['src_seq']).cpu().detach().numpy()).tolist())) ) #" ".join(func(torch.stack(batch['src_seq']).detach().cpu().numpy()).tolist())
            target_text = " ".join( list(map(lambda x: x[0], func(torch.stack(batch['tgt_seq']).cpu().detach().numpy()).tolist())) )  #batch['tgt_seq'][0]
            
            model_out_text = " ".join( list(filter(lambda x: x != "SOS", func(model_out.detach().cpu().numpy()).tolist())) ) #Turn tokens back to notes and filters out padding
            #If this is empty that means it's all SOS for some reason
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            #Print to console
            print_msg('-'*console_width) #apparently using this instead of print doesnt mess up the tqdm loading bar
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break

def get_ds(config):
    # Load dataset; tokenizer is already defined from dictionary
    ds_raw = createDataset(config["num_files"])
    
    #Tokenize all the inputs
    for i in range(0, len(ds_raw)):
        ds_raw[i][0] = list(map(lambda x: tokenizer[x], ds_raw[i][0]))
        ds_raw[i][1] = list(map(lambda x: tokenizer[x], ds_raw[i][1]))

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BarDataset(train_ds_raw, tokenizer, config['seq_len'])
    val_ds = BarDataset(val_ds_raw, tokenizer, config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        max_len_src = max(max_len_src, len(item[0]))
        max_len_tgt = max(max_len_tgt, len(item[1]))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(dataset=train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=lambda x: x)
    val_dataloader = DataLoader(dataset=val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader #Returns formatted tensors

def get_model(config, vocab_src_len, vocab_tgt_len): #model from model.py
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader = get_ds(config)
    model = get_model(config, len(list(tokenizer.keys())), len(list(tokenizer.keys()))).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict']) 
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer["PAD"], label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            for pair in batch:
                encoder_input = pair['encoder_input'].to(device) # (b, seq_len)
                decoder_input = pair['decoder_input'].to(device) # (B, seq_len)
                encoder_mask = pair['encoder_mask'].to(device) # (B, 1, 1, seq_len)
                decoder_mask = pair['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
                encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
                proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
                label = pair['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
                loss = loss_fn(proj_output.view(-1, len(list(tokenizer.keys()))), label.view(-1))
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
                writer.add_scalar('train loss', loss.item(), global_step)
                writer.flush()

            # Backpropagate the loss
                loss.backward()

            # Update the weights
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, config['seq_len'], device, lambda msg: batch_iterator.write(msg), 1)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

def nObj(noteData):
    return music21.note.Note(music21.pitch.Pitch(noteData[0]), type=durationCodes[noteData[1]])

def cObj(chordData):
    pitches = [music21.pitch.Pitch(x) for x in chordData[0]]
    return music21.chord.Chord(pitches, type=durationCodes[chordData[1]])

def decodeNotes(notes=list):
    lastNote = [-1, -1, -1]
    lastChord = [[], -1, -1] #Pitch duration beat
    outputStream = music21.stream.Stream()
    barCounter = 0

    for note in notes:
        noteData = note.split("-") #[pitch, duration, beat]
        noteData = [ float(x) for x in noteData ]
        if noteData[2] < lastNote[2]: #This means a new bar is starting
            barCounter +=1
            if len(lastChord[0]) > 0:
                outputStream.insert(((barCounter*4)+(lastChord[2]/8)), cObj(lastChord))
                lastChord = [[], 0, 0]
            outputStream.insert(((barCounter*4)+(noteData[2]/8)), nObj(noteData))
            lastNote = noteData #Last beat is now the current beat
        elif noteData[2] == lastNote[2]: #Must mean a new chord 
            if len(lastChord[0]) == 0: #Add the previous note as well to chord
                lastChord[0].append(lastNote[0])
                lastChord[1] = lastNote[1]
                lastChord[2] = lastNote[2]
            if noteData[0] not in lastChord[0]: #Unique note to be added to chord
                lastChord[0].append(noteData[0])
                if lastChord[1] < noteData[1]: #Always go with longest chord out of notes
                    lastChord[1] == noteData[1] 
        else: #Chord ended, and a new bar has not started
            if len(lastChord[0]) > 0:
                outputStream.insert(((barCounter*4)+(lastChord[2]/8)), cObj(lastChord)) #Add chord to measure stream
                lastChord = [[], 0, 0] #Change lastChord to empty again
            lastNote = noteData
            outputStream.insert(((barCounter*4)+(noteData[2]/8)), nObj(noteData))
    outputStream.show("midi") #midi for audio, text for notes, none for musescore(if available)
    return outputStream

def getOutput(filename): #filename of midi file to send into transformer
    config = get_config()
    #Get device information
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    device = torch.device(device)
    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    #Load the model and latest weights
    model = get_model(config, len(list(tokenizer.keys())), len(list(tokenizer.keys()))).to(device)
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None

    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        print("Successfully loaded model")
    else:
        print('No model to preload')
        exit()
    
    predicted = []
    ds = arrangeFile(filename)
    model.eval()
    for i in tqdm(range(0, len(ds)),desc="Obtaining output..."):
        with torch.no_grad():
            model_out = greedy_decode(model, ds[i]["encoder_input"], ds[i]["encoder_mask"], config["seq_len"], device)
            func = np.vectorize(lambda x: list(tokenizer.keys())[x])
            model_out_text = list(filter(lambda x: x != "SOS", func(model_out.detach().cpu().numpy()).tolist())) #Turn tokens back to notes and filters out padding
            predicted = predicted + model_out_text
    print(" ".join(predicted))
    return decodeNotes(predicted)

#Begin training loop
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)

#Get an output from a filename
#filename = r"C:\Users\alexm\Downloads\Sound Testing 2\midi songs\C_major_scale.mid"
#getOutput(filename)