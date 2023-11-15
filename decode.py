#File that you can copy paste the raw input from train.py into "rawInput" and get the output
import music21
from music21 import converter
import numpy as np
import pretty_midi
#file = r"C:\Users\alexm\Downloads\Sound Testing 2\midi songs\C_major_scale.mid"
file = r"C:\Users\alexm\Downloads\Sound Testing 2\midi songs\Debussy_Reverie.mid"

rawInput = "66-2-8 59-2-8 71-2-8 51-2-8 48-4-8 57-2-12 55-2-12 54-2-16 45-2-16 55-2-20 57-2-22 55-2-22 75-8-24 76-4-31 47-4-26 55-4-26 54-2-31 48-4-31 52-4-31 57-2-26 64-32-8 76-16-8 54-2-8 55-2-10 57-8-12 50-16-22 49-4-13 64-2-18 65-4-22 73-2-29 59-8-31 57-2-27 74-2-27 74-8-31 64-2-31 62-2-31 76-16-8 56-2-10 64-2-10 72-2-14 57-2-14 71-2-16 55-2-16 72-4-19 53-4-19 69-4-19 74-2-24 52-2-24 72-2-24 71-4-28 47-2-31 72-2-31 74-2-31 72-2-31 68-2-31 75-2-31 74-2-31 50-2-28 52-2-31 72-2-31 50-4-8 59-2-8 74-4-8 71-2-12 71-2-16 67-2-20 69-2-22 57-2-24 66-2-24 67-2-27 64-2-29 66-2-31 59-4-31 45-4-14 63-2-31 74-2-31 69-2-14 72-2-31 72-16-14 52-1-8 72-1-8 72-2-8 47-2-11 71-2-11 72-2-14 69-2-16 71-2-18 67-2-21 66-2-26 64-2-28 60-4-31 59-2-31 48-8-21 69-8-30 64-4-21 66-2-31 57-2-31 74-4-8 63-4-8 66-2-16 69-2-22 59-8-28 66-8-28 63-2-31 59-8-8 67-4-18 64-4-18 67-4-24 63-2-31 61-2-31 67-2-13 52-8-18 47-2-31 69-4-8 67-4-12 66-4-12 64-16-16 60-2-31 45-2-31 59-4-31 48-2-16 57-2-21 56-2-24 57-4-26 62-2-30 47-2-30 47-2-31 63-4-8 47-8-8 40-2-8 55-2-8 54-2-10 52-16-13 55-4-31 54-2-31 55-4-13 59-2-20 55-2-20 55-2-27 57-2-29 50-2-29 59-8-31 60-2-18 57-2-18 57-4-22 54-4-22 52-2-31 50-4-31 56-4-10 52-2-16 60-8-19 57-8-19 52-8-29 54-2-14 48-4-19 47-2-24 59-4-29 55-2-31 45-2-26 55-2-31 54-2-31 64-2-31 49-2-31 59-8-8 50-8-8 54-2-12 54-8-16 46-8-16 62-16-26 52-2-13 62-2-21 47-4-26 55-2-30 54-2-31 52-4-31 44-4-31 61-4-22 55-4-8 64-16-8 49-2-8 61-2-8 61-16-13 59-2-28 59-8-31 59-4-10 54-2-18 52-4-20 50-8-24 47-8-24 52-2-31 46-4-13 58-2-30 43-4-31 62-2-8 52-4-8 44-4-8 50-2-8 49-8-10 58-8-20 54-8-20 42-8-20 52-2-31 40-4-11 61-2-16 59-2-18 66-2-24 67-8-30 62-8-30 47-4-30 50-2-31 59-4-8 49-8-8 52-8-8 66-2-18 50-2-18 59-8-22 66-2-31 52-2-31 50-2-31 67-2-31 49-2-31 40-2-31 57-4-8 66-2-13 64-2-16 50-2-22 49-2-24 50-4-27 47-4-27 71-16-27 50-2-10 52-2-12 50-2-12 70-8-14 67-2-29 66-2-31 66-2-31 42-4-13 66-4-13 47-8-24 43-8-24 44-4-31 52-2-16 64-2-19 49-2-19 71-16-24 50-4-16 62-4-24 64-2-31 62-2-31 71-2-8 64-4-8 45-4-12 64-2-16 57-2-16 55-2-19 69-8-21 71-4-31 69-4-31 60-16-12 52-2-26 51-8-30 54-2-20 55-2-24 54-4-27 47-8-30 59-8-30 44-4-8 71-8-8 62-1-8 52-16-8 71-8-28 67-4-8 69-2-13 67-2-16 67-2-22 51-4-28 52-2-31 40-2-31 66-4-18 69-4-24 59-2-31 60-2-31 47-32-8 59-16-8 69-2-11 69-16-16 66-4-31 68-4-12 59-2-20 59-2-24 52-2-29 67-2-29 64-4-31 57-2-22 67-4-31 60-8-8 69-2-8 67-2-8 66-2-12 67-2-16 69-2-16 67-2-20 66-4-29 57-2-31 63-4-20 59-4-20 67-4-20 48-4-31 56-2-31 69-2-21 64-8-31 47-8-21 64-4-8 57-4-10 45-4-16 55-2-24 54-4-27 57-2-31 55-2-31 47-2-14 47-16-19 71-16-19 63-8-19 55-2-29 64-8-29 64-8-8 54-4-8 57-4-18 63-4-18 69-4-24 52-8-30 55-4-12 71-4-26 57-2-31 56-8-30 72-8-30 64-8-30 74-2-31 47-16-8 71-8-8 64-8-8 72-2-10 56-8-13 63-8-24 71-2-12 72-2-12 71-2-16 72-2-16 71-2-18"
rawInput = rawInput.split(" ")

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
BEATS = list(range(0,32)) # Since smallest duration is 32nd note split each bar into 32 slots

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

def nObj(noteData):
    return music21.note.Note(music21.pitch.Pitch(noteData[0]), type=durationCodes[noteData[1]])

def cObj(chordData):
    pitches = [music21.pitch.Pitch(x) for x in chordData[0]]
    return music21.chord.Chord(pitches, type=durationCodes[chordData[1]])

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


encoded_notes = midi_to_notes(file)
decodeNotes(encoded_notes).show() #"midi", "text" encoded_notes from midi file