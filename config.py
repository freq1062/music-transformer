from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "num_files":2, #Only for debugging(all the files are saved to midiData.pkl)
        "lr": 10**-4,
        "seq_len": 350, #Actually 347 because of SOS, EOS tokens
        "d_model": 512, #Dimensionality of embedding
        "datasource": 'maestro', #Source of data
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest", #Load the latest model weights available
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])