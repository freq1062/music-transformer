from model3 import MusicTransformer
from dataset import EventDataset
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import warnings
from tqdm import tqdm
import pathlib
from pathlib import Path
import numpy as np

from torch.utils.tensorboard import SummaryWriter

#configure.run() #Music21 visualization
import tensorflow as tf
import glob
import pickle
import pretty_midi
import music21

def getFilepaths(custom:str="maestro"):
    if custom == "maestro":
        data_dir = pathlib.Path('data/maestro-v2.0.0')
        if not data_dir.exists():
            tf.keras.utils.get_file(
            'maestro-v2.0.0-midi.zip',
            origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
            extract=True,
            cache_dir='.', cache_subdir='data',
        )
        return glob.glob(str(data_dir/'**/*.mid*'))
    else:
        data_dir = pathlib.Path(custom)
        return glob.glob(str(data_dir/'**/*.mid*'))

def closest(lst, K):return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

DURATIONS = [1,2,4,8,16,32] #Possible durations starting with 1 = 32nd note. 32nd, 16th, 8th, quarter, half, whole
durationCodes = { #For the decoder
    1: "32nd",
    2: "16th",
    4: "eighth",
    8: "quarter",
    16: "half",
    32: "whole"
}

def midi_to_events(midi_file: str): #Convert a midi file into untokenized events
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    events = []
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prevstart = 0
    for element in sorted_notes:
        start = element.start
        if start-prevstart > 0 and prevstart != 0: #Some beat passed
            events.append(f"t{closest(DURATIONS, (start-prevstart)*16)}") #time pass event
        events.append(f"{element.pitch}-{(closest(DURATIONS, (element.duration)*16))}") #note event: pitch-duration
        prevstart = start
    return events

def buildVocab(words,specialTokens=[]): #From a list of words create a dictionary "VOCABULARY" of {"element": int}
    words = words+specialTokens
    VOCABULARY = dict()
    counter = 0
    for ele in words:
        #print(ele)
        if ele not in list(VOCABULARY.keys()):
            VOCABULARY[str(ele)] = counter
            counter+=1
    return VOCABULARY

def createDataset(filenames, pkl:bool=True): #Recheck this
    if pkl: #load from pkl files
        with open(filenames[0], 'rb') as fp: ds = pickle.load(fp)[:1024] #dataset
        with open(filenames[1], 'rb') as fp: tokenizer = pickle.load(fp) #tokenizer
        return tokenizer, ds
    else:
        ds = []
        seq_len = get_config()["seq_len"]-2 #EOS, SOS tokens
        #Actually seq_len depends on what was saved to the pkl file if loading that way
        notes_raw = []
        for f in tqdm(filenames[:],desc=f"Loading files from dataset..."): #Later: Get data from premade .pkl file
            #Combine all of the notes from files into 1 array ["note1", "note2", "note3"...]
            fileNotes = midi_to_events(f)
            splitNotes = [fileNotes[i:i + seq_len] for i in range(0, len(fileNotes), seq_len)]
            notes_raw = notes_raw + fileNotes
            for j in range(0, len(splitNotes)-1): ds.append([splitNotes[j], splitNotes[j+1]]) #split into [ [[bar1], [bar2]], [[bar2],[bar3]]... ]

        tokenizer = buildVocab(notes_raw, ["SOS", "EOS", "PAD"])
        return tokenizer, ds

tokenizer, ds = createDataset([r"training_data\midiData_EVENTS.pkl", r"training_data\tokens_EVENTS.pkl"], True)

def saveData(filenames:str="maestro"): #From the raw dataset creates the pkl files which load faster
    tokenizer, ds = createDataset(getFilepaths(filenames), pkl=False)
    with open(r'tokens_EVENTS.pkl', 'wb') as fp:
        pickle.dump(tokenizer, fp)
        print("Saved tokens to file")
    with open(r"midiData_EVENTS.pkl", 'wb') as fp:
        pickle.dump(ds, fp)
        print("Saved training data to file")

#saveData()
#exit()

#Decoding methods

def top_k_decode(model, input, config, device, k=3):
    sos_idx = tokenizer['SOS']
    eos_idx = tokenizer['EOS']

    #encoder_output = model.encode(source, source_mask)
    output = torch.empty(1, 1).fill_(sos_idx).type_as(input).to(device)

    while True:
        if output.size(1) == config["seq_len"]:
            break

        #decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model(input.unsqueeze(0))
        #out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get top-k words and their probabilities
        #prob = model.project(out[:, -1])
        _, top_k_words = torch.topk(out[:, -1], k=k, dim=1)

        # Sample a word from the top-k words
        next_word = top_k_words[:, torch.randint(k, (1,), dtype=torch.long)].squeeze()

        output = torch.cat(
            [output, torch.empty(1, 1).type_as(input).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return output.squeeze(0)

#

def run_validation(model, validation_ds, device, print_msg, num_examples:int=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count+=1

            input = batch["input"].squeeze(0).to(device)
            print("Decoding sample output")
            model_out = top_k_decode(model, input, get_config(), device, k=3)
            func = np.vectorize(lambda x: list(tokenizer.keys())[x])

            #Convert source and target tokens to text
            source_text = " ".join( list(map(lambda x: x[0], func(torch.stack(batch['src_seq']).cpu().detach().numpy()).tolist())) ) 
            target_text = " ".join( list(map(lambda x: x[0], func(torch.stack(batch['tgt_seq']).cpu().detach().numpy()).tolist())) ) 
            model_out_text = " ".join( list(filter(lambda x: x != "PAD", func(model_out.detach().cpu().numpy()).tolist())) ) 

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
            

def arrangeFile(filename): #Arrange an input midi file to the same format as the one the model recognizes
    ds = []
    seq_len = get_config()["seq_len"]-2
    notes_raw = []
    fileNotes = midi_to_events(filename)
    
    notes_raw = list(map(lambda x: tokenizer[x], fileNotes)) #Tokenize everything
    notes_raw = np.split(notes_raw, np.arange(seq_len, len(notes_raw), seq_len)) #[[seq_len], [seq_len], ...]

    #Create special tokens for padding
    sos_token = torch.tensor([tokenizer["EOS"]], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer["SOS"]], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer["PAD"]], dtype=torch.int64)

    for seq in notes_raw:
        num_padding_tokens = seq_len - len(seq)
        input = torch.cat([
            sos_token, #1 additional
            torch.tensor(seq, dtype=torch.int64),
            eos_token, #2 additional
            torch.tensor([pad_token] * num_padding_tokens, dtype=torch.int64),])
        #encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int()
        ds.append({"input":input, "src_seq":seq})
    return ds

def nObj(noteData):
    return music21.note.Note(music21.pitch.Pitch(noteData[0]), type=durationCodes[noteData[1]])

def cObj(chordData):
    pitches = [music21.pitch.Pitch(x) for x in chordData[0]]
    return music21.chord.Chord(pitches, type=durationCodes[chordData[1]])

def decodeEvents(events=list):
    outputStream = music21.stream.Stream()
    lastEvent = "t0"
    lastChord = {"pitches":[], "duration":0} #pitches, duration
    globaltime = 0
    for event in events:
        if not event.startswith("t"):
            pitch, duration = [float(x) for x in event.split("-")]
        if lastEvent == "note": #2 notes in the same time means chord
            lastChord["pitches"].append(pitch)
            if duration > lastChord.duration: #max duration for chord
                lastChord.duration = duration
        elif lastEvent.startswith("t"): #Timestep
            if lastEvent != "t0":
                globaltime += int(lastEvent.replace("t",""))/8 #Add globaltime
                if len(lastChord["pitches"]) > 0:
                    outputStream.insert(globaltime, cObj(lastChord["pitches"], lastChord.duration))
                else:
                    outputStream.insert(globaltime, nObj([pitch, duration]))
        lastEvent = event
    return outputStream

def notesToTensor(notes_raw): #arr
    notes_raw = list(map(lambda x: tokenizer[x], notes_raw))
    seq_len = get_config()["seq_len"]
    sos_token = torch.tensor([tokenizer["EOS"]], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer["SOS"]], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer["PAD"]], dtype=torch.int64)

    enc_num_padding_tokens = seq_len - len(notes_raw) - 2
    encoder_input = torch.cat([
        sos_token,
        torch.tensor(notes_raw, dtype=torch.int64),
        eos_token,
        torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64),])
    encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int()
    return {"encoder_input": encoder_input, "encoder_mask":encoder_mask}

def get_ds(config):
    # Load dataset; tokenizer is already defined from dictionary
    ds_raw = ds
    tokens = list(tokenizer.keys())
    #Tokenize all the inputs
    for i in range(0, len(ds_raw)):
        ds_raw[i][0] = list(map(lambda x: tokenizer.get(x) if x in tokens else None, ds_raw[i][0]))
        list(filter(lambda x: x != None, ds_raw[i][0]))
        ds_raw[i][1] = list(map(lambda x: tokenizer.get(x) if x in tokens else None, ds_raw[i][1]))
        list(filter(lambda x: x != None, ds_raw[i][1]))

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = EventDataset(train_ds_raw, tokenizer, config['seq_len'])
    val_ds = EventDataset(val_ds_raw, tokenizer, config['seq_len'])

    #print(np.shape(train_ds[0]["decoder_mask"]))
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

def get_model(config, vocab_size): 
    model = torch.jit.script(MusicTransformer(vocab_size, config["d_model"], 8))
    #Actually seq_len doesn't really matter because it's all done in preprocessing
    #Currently d_model is 512, num_heads is 8 according to original transformer paper
    #Leaving depth, d_ff, dropout to default 
    return model

def train_model(config):
    #Define device to be used
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    device = torch.device(device)
    #Check if weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader = get_ds(config) #Run validation?
    model = get_model(config, len(list(tokenizer.keys()))).to(device)
    #Tensorboard stuff
    writer = SummaryWriter(config["experiment_name"])

    #Hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer["PAD"], label_smoothing=0.1).to(device)

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
    
    #Training loop
    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train() #Put into training mode
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            inputs = []
            labels = []

            for pair in batch:
                inputs.append(pair["input"].to(device)) #(seq_len)
                labels.append(pair["label"].to(device)) #(batch, seq_len)

            inputs = torch.stack(inputs)
            labels = torch.stack(labels)
            model_output = model(inputs)
            #Calculate cross entropy loss
            loss = loss_fn(model_output.view(-1, len(list(tokenizer.keys())) ), labels.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            #Backpropagate
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
                
            global_step += 1
        
        #Run validation
        run_validation(model, val_dataloader, device, lambda msg: batch_iterator.write(msg), 1)

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

def getOutput(filename=str): #filename of midi file to send into transformer, mode, number of bars to generate
    config = get_config()
    #Get device information
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    device = torch.device(device)
    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    #Load the model and latest weights
    model = get_model(config, len(list(tokenizer.keys()))).to(device)
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None

    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        print("Successfully loaded model")
    else:
        print('No model to preload')
        return
    
    model.eval()
    arranged_file = arrangeFile(filename)[0] #Only up to max seq_len for now
    with torch.no_grad():
        model_out = top_k_decode(model, arranged_file["input"], config, device, 3)
        #Detokenize model output
        func = np.vectorize(lambda x: list(tokenizer.keys())[x])
        model_out_text = list(filter(lambda x: x != "SOS", func(model_out.detach().cpu().numpy()).tolist())) #Turn tokens back to notes and filters out padding
        print(model_out_text)
        return decodeEvents(model_out_text)

#Begin training loop
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()    
    train_model(config)
        
#filename = r""
#getOutput(filename).show()
