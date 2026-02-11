import mido
from torch.utils.data import Dataset
import torch
import os
import random

class MIDITokenizer:
    def __init__(self):
        # Token ranges
        self.note_on_base = 0
        self.note_off_base = 128
        self.time_shift_base = 256
        self.velocity_base = 356

        self.time_shift_bins = 100 # up to 1 second (100 * 10ms)
        self.velocity_bins = 32

        self.pad_id = 388
        self.sos_id = 389
        self.eos_id = 390

        self.vocab_size = 391

    # ---------------- ENCODE ---------------- #
    def midi_to_events(self, midi_path: str) -> list[int]:
        mid = mido.MidiFile(midi_path)
        events = []

        all_msgs = []
        for track in mid.tracks:
            time = 0
            for msg in track:
                time += msg.time
                all_msgs.append((time, msg))

        all_msgs.sort(key=lambda x: x[0])
        last_time = 0

        for time, msg in all_msgs:
            delta_ticks = time - last_time
            delta_sec = mido.tick2second(delta_ticks, mid.ticks_per_beat, 500000)
            delta_ms = int(delta_sec * 100)

            # Time shifts
            while delta_ms > 0:
                shift = min(delta_ms, self.time_shift_bins)
                events.append(self.time_shift_base + shift)
                delta_ms -= shift

            # Notes
            if msg.type == "note_on" and msg.velocity > 0:
                vel_bin = min(msg.velocity // 4, 31)
                events.append(self.velocity_base + vel_bin)
                events.append(self.note_on_base + msg.note)

            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                events.append(self.note_off_base + msg.note)

            last_time = time

        return events

    # ---------------- DECODE ---------------- #
    def events_to_midi(self, events: list[int], out_path="out.mid"):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        current_time = 0
        current_velocity = 64

        for tok in events:
            if tok in (self.pad_id, self.sos_id, self.eos_id):
                continue

            # Note On
            if 0 <= tok < 128:
                track.append(
                    mido.Message(
                        "note_on",
                        note=tok,
                        velocity=current_velocity,
                        time=current_time,
                    )
                )
                current_time = 0

            # Note Off
            elif 128 <= tok < 256:
                track.append(
                    mido.Message(
                        "note_off",
                        note=tok - 128,
                        velocity=0,
                        time=current_time,
                    )
                )
                current_time = 0

            # Time Shift
            elif 256 <= tok < 356:
                shift = tok - 256
                delta_sec = shift * 0.01
                delta_ticks = int(mido.second2tick(delta_sec, mid.ticks_per_beat, 500000))
                current_time += delta_ticks

            # Velocity
            elif 356 <= tok < 388:
                vel_bin = tok - 356
                current_velocity = int(vel_bin * 4)

        mid.save(out_path)
        return out_path

class MaestroDataset(Dataset):
    def __init__(self, midi_folder, seq_len=3000):
        self.midi_files = [os.path.join(midi_folder, f) for f in os.listdir(midi_folder) if f.endswith('.mid')]
        self.seq_len = seq_len
        self.tokenizer = MIDITokenizer() # Your MIDI tokenizer

    def __len__(self):
        # We can define an 'epoch' as passing through each file once
        return len(self.midi_files)

    def __getitem__(self, idx):
        # 1. Load a random MIDI file
        # 2. Convert to events
        # 3. Pick a RANDOM window of seq_len
        file_path = self.midi_files[idx]
        events = self.tokenizer.midi_to_events(file_path)
        
        if len(events) <= self.seq_len:
            # Pad if too short
            events += [self.tokenizer.pad_id] * (self.seq_len - len(events) + 1)
        
        # Pick a random starting point for the 3000-token window
        start_idx = random.randint(0, len(events) - self.seq_len - 1)
        chunk = events[start_idx : start_idx + self.seq_len + 1]
        
        input_seq = torch.tensor(chunk[:-1], dtype=torch.long)
        label_seq = torch.tensor(chunk[1:], dtype=torch.long)
        
        return {"input": input_seq, "label": label_seq}