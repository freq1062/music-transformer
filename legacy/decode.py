#File that you can copy paste the raw input from train.py into "rawInput" and get the output
import music21
from music21 import converter
import numpy as np
import pretty_midi
import pandas as pd
import matplotlib as plt
import collections
file = "" #filepath

rawInput = "t2 75-2 t2 71-2 t8 78-16 t8 69-2 t2 73-2 t2 75-2 t2 76-16 t2 71-2 t2 68-2 t8 75-16 t16 76-8 t8 80-4 t4 83-4 t4 75-16 t16 73-4 t4 76-4 t4 80-4 t4 71-16 t16 69-4 t4 73-4 t4 76-4 t4 68-16 t16 69-2 t2 73-2 t2 76-4 t4 80-4 t4 78-4 t4 75-8 t8 68-8 t16 66-4 t4 69-4 t4 73-4 t4 64-16 t16 66-2 t2 69-2 t2 73-2 t2 76-2 t2 74-2 t2 68-8 t8 66-16 t16 62-8 t32 66-32 t8 57-2 t2 59-2 t2 61-2 t2 59-2 t2 57-2 t2 59-2 t2 61-2 t2 59-2 t2 57-2 t2 61-16 t8 66-8 t8 69-8 63-32 t8 66-8 t8 71-8 t8 68-8 t8 71-8 64-16 t8 68-8 t8 69-16 73-32 t16 61-2 t2 64-2 t2 66-2 t2 69-2 t2 68-2 t2 66-2 t2 64-16 t8 66-8 69-8 t8 68-8 71-8 t8 69-8 73-8 t8 69-16 73-32 t16 61-2 t2 64-2 t2 66-2 t2 69-2 t2 68-2 t2 66-2 t2 64-16 t8 66-2 69-8 t2 61-2 t2 64-2 t2 68-2 71-8 t2 63-2 t2 66-2 t2 69-2 73-8 t2 64-2 t2 68-2 t2 71-2 75-8 t2 66-2 t2 69-2 t2 73-2 76-8 t2 68-2 t2"
rawInput = list(filter(lambda x: x != "SOS", rawInput.split(" ")))
#rawInput = rawInput.split(" ")

#Take the output of the transformer and turn it into a midi file
DURATIONS = [1,2,4,8,16,32] #Possible durations starting with 1 = 32nd note. 32nd, 16th, 8th, quarter, half, whole
durationCodes = {
    1: "32nd",
    2: "16th",
    4: "eighth",
    8: "quarter",
    16: "half",
    32: "whole"
}
BEATS = list(range(0,16)) # Since smallest duration is 32nd note split each bar into 32 slots

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

def nObj(noteData): #noteData:[pitch, durationCode]
    return music21.note.Note(music21.pitch.Pitch(noteData[0]), type=durationCodes[noteData[1]])

def cObj(chordData):
    pitches = [music21.pitch.Pitch(x) for x in chordData[0]]
    return music21.chord.Chord(pitches, type=durationCodes[chordData[1]])

def decodeEvents(events):
    outputStream = music21.stream.Stream()
    lastEvent = "t0"
    lastChord = {"pitches":[], "duration":0} #pitches, duration
    globaltime = 0
    for event in events:
        if not event.startswith("t"): pitch, duration = [float(x) for x in event.split("-")]
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

decodeEvents(rawInput).show("midi")

def decodeNotes(notes):
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
    
    return outputStream

def midi_to_notes(midi_file: str) -> pd.DataFrame:
  pm = pretty_midi.PrettyMIDI(midi_file)
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list)

  # Sort the notes by start time
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  #instrument.notes #sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  for note in instrument.notes:
    start = note.start*2
    end = note.end*2
    notes['pitch'].append(note.pitch)
    notes['start'].append(start)
    notes['end'].append(end)
    notes['step'].append(start - prev_start)
    notes['duration'].append(closest(DURATIONS, float(end - start)))
    prev_start = start

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})
  
def plot_piano_roll(notes: pd.DataFrame, count: int):
  if count:
    title = f'First {count} notes'
  else:
    title = f'Whole track'
    count = len(notes['pitch'])
  plt.figure(figsize=(20, 4))
  plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
  plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
  plt.plot(
      plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
  plt.xlabel('Time [s]')
  plt.ylabel('Pitch')
  _ = plt.title(title)
  plt.show()

encoded_notes = midi_to_notes(file)#
decodeNotes(rawInput).show("midi")
 #"midi", "text" encoded_notes from midi file
