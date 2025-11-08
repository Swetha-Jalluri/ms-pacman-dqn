import os, yaml, random, numpy as np, torch

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_device(pref="mps"):
    # prefer Apple Metal (mps) on Mac, else cuda, else cpu
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
