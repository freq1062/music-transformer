from pathlib import Path

def get_config():
    return {
        "batch_size": 128,
        "num_epochs": 200, #Usually done by 200
        "num_files":1282, #Number of files from the raw midi files, max 1282
        "lr": 10**-4, #For the mask
        "seq_len": 200, #Actually will be seq_len - 2 because of EOS, SOS tokens
        "d_model": 512, #Dimensionality of embedding
        "datasource": 'maestro', #Source of data
        "model_folder": "weights", #Actually maestro\weights
        "model_basename": "tmodel_",
        "num_samples":2048, #if from pkl file determines number of pairs to train on
        "preload": "latest", #Load the latest model weights available
        "experiment_name": "runs/tmodel" #Folder for tensorboard + other metric stuff
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    print("Loading latest weights")
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
