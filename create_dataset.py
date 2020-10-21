import torch
import numpy as np
import torch.utils.data.dataset as Dataset
import os


class BluekeepDataset (Dataset.__class__):
    def __init__(self, root_dir,eos_file):
        self.root_dir = root_dir
        self.streams = os.listdir(root_dir)
        self.exp_or_safe = np.load(os.path.join(root_dir, eos_file))
        self.transform = None

    def __len__(self):
        return len(self.streams)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif isinstance(idx, np.ndarray):
            idx = idx.tolist()
        elif isinstance(idx, np.matrix):
            idx = np.asarray(idx)
            idx = idx.tolist()
        elif not(isinstance(idx, list)):
            raise Exception("Please use one of the following data-type: list, np.array / matrix, torch.tensor")

        packet_id = os.path.join(self.root_dir, self.streams[idx])
        packet = np.load(packet_id)
        exp_or_safe = self.exp_or_safe[idx]
        sample = {'packet': packet, 'EoS': exp_or_safe}

        return sample

def create_dataset(ROOT, EOS):
    stream_dataset = BluekeepDataset(root_dir=ROOT, eos_file=EOS)
    return stream_dataset