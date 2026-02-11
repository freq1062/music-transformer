import os
import glob
import torch
import random
import kagglehub
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import utils
import model4

# ==========================================
# CONFIGURATION - CHANGE PER MACHINE
# ==========================================
SHARD_ID = 0 # 0-9
TOTAL_SHARDS = 10
EPOCHS = 50
BATCH_SIZE = 16
SEQ_LEN = 1024 # According to paper
LEARNING_RATE = 1e-4
DEPTH = 6

# ==========================================
# DATA GETTING & SHARDING
# ==========================================
def get_sharded_paths(shard_id, total_shards):
    data_dir = os.path.expanduser("~/music-transformer/maestro-v3.0.0")
    
    # Search recursively for .mid or .midi files
    all_midi_files = glob.glob(os.path.join(data_dir, "**/*.mid*"), recursive=True)
    all_midi_files.sort()
    
    files_per_shard = len(all_midi_files) // total_shards
    start = shard_id * files_per_shard
    end = start + files_per_shard if shard_id != total_shards - 1 else len(all_midi_files)
    
    my_files = all_midi_files[start:end]
    print(f"Machine {shard_id} found {len(my_files)} files out of {len(all_midi_files)}")
    return my_files

# ==========================================
# DATASET CLASS
# ==========================================
class MaestroShardDataset(Dataset):
    def __init__(self, file_list, tokenizer, seq_len=3000):
        self.file_list = file_list
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.file_list) * 5  # Virtual length to see more random windows

    def __getitem__(self, idx):
        # Pick a random file from our shard
        f_path = random.choice(self.file_list)
        try:
            events = self.tokenizer.midi_to_events(f_path)
            if len(events) <= self.seq_len:
                events += [0] * (self.seq_len - len(events) + 1)
            
            start_idx = random.randint(0, len(events) - self.seq_len - 1)
            chunk = events[start_idx : start_idx + self.seq_len + 1]
            
            return {
                "input": torch.tensor(chunk[:-1], dtype=torch.long),
                "label": torch.tensor(chunk[1:], dtype=torch.long)
            }
        except Exception:
            return self.__getitem__(random.randint(0, len(self.file_list)-1))

# ==========================================
# TRAINING LOOP WITH LOGGING
# ==========================================
def log_to_file(message):
    with open(f"shard_{SHARD_ID}_log.txt", "a") as f:
        f.write(message + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = utils.MIDITokenizer()

model = model4.MusicTransformer(
    vocab_size=391, # 388 events + SOS, EOS, PAD
    d_model=512,
    num_heads=8,
    depth=DEPTH,
    max_len=SEQ_LEN,
    dropout=0.1
).to(device)

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_files = get_sharded_paths(SHARD_ID, TOTAL_SHARDS)
    
    dataset = MaestroShardDataset(my_files, tokenizer, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_loss = float("inf")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Shard {SHARD_ID} | Epoch {epoch}")
        
        for batch in pbar:
            inputs, labels = batch["input"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.transpose(1, 2), labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        msg = f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}"
        print(msg)
        log_to_file(msg)
        
        # Save shard-specific checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'shard': SHARD_ID,
                'loss': avg_loss
            }, f"model_shard_{SHARD_ID}.pt")

if __name__ == "__main__":
    run_training()