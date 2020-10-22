import numpy as np
from torch.utils.data import Dataset
import os


class BluekeepDataset (Dataset):
    def __init__(self, root_dir, eos_file, transform=None):
        self.root_dir = root_dir
        self.streams = os.listdir(root_dir)
        self.exp_or_safe = np.load(os.path.join(root_dir, eos_file))
        self.transform = transform

    def __len__(self):
        return len(self.streams)

    def __getitem__(self, idx):
        if isinstance(idx, np.matrix):
            idx = np.asarray(idx)
            idx = idx.tolist()
        elif isinstance(idx, np.ndarray):
            idx = idx.tolist()
        elif not(isinstance(idx, list)) and not(isinstance(idx, int)):
            raise Exception("Please use one of the following data-type: list, np.array / matrix, torch.tensor")

        stream_id = os.path.join(self.root_dir, self.streams[idx])
        stream = np.load(stream_id)
        stream_eos = np.zeros(2)    # This place creates problems
        stream_eos[self.exp_or_safe] = 1
        stream_eos = stream_eos.astype('float')
        sample = {'stream': stream, 'EoS': stream_eos}

        if self.transform:
            sample = self.transform(sample)

        return sample


def create_dataset(root_dir, eos):
    stream_dataset = BluekeepDataset(root_dir=root_dir, eos_file=eos)
    return stream_dataset
