from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

f_type = torch.float32

multi_get = lambda d, keys: [d.get(key) for key in keys]
accuracy = lambda p, y: ((p > .5) == y).numpy().mean()

class LDADataset(Dataset):
    def __init__(self, W, Y):
        super(LDADataset, self).__init__()
        self.W, self.Y = W, Y
        # Metadata
        self.I, self.V = self.W.shape
        self.patient_idxs = torch.arange(self.I)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return [self.W[idx], self.Y[idx], self.patient_idxs[idx]]


def collate_fn_tensor(list_of_WYI_tuples):
    Wb = torch.stack([W for W, Y, I in list_of_WYI_tuples], dim=0)
    Yb = torch.stack([Y for W, Y, I in list_of_WYI_tuples], dim=0)
    Ib = torch.stack([I for W, Y, I in list_of_WYI_tuples], dim=0)
    return Wb, Yb, Ib


def get_path_data(data_root, split):
    from pathlib import Path
    assert split in ['val', 'test', 'train']
    return Path(f"{data_root}_{split}.pkl")


def get_data(path):
    import pickle
    data = pickle.load(open(path, "rb"))
    return data


def visualize_topics(topics, fname, ax=None):
    from matplotlib import pyplot as plt
    K, V = topics.shape
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(topics, cmap="Greys")
        ax.set_xlabel("Words / Concepts")
        ax.set_ylabel("Topics")
        ax.set_title(f"{K} Topics")
        fig.savefig(fname)
        print(f"Saved topics to {fname}")
    else:
        ax.imshow(topics, cmap="Greys", vmin=0, vmax=.5)
        ax.set_xlabel("Words / Concepts")

def visualize_coefficients(eta, ax=None):
    ax.imshow(eta, cmap="bwr")

def check_and_load_data_LDA(split, dataroot, dataname, batch_size):
    data_path = get_path_data(Path(dataroot) / dataname, split)
    assert data_path.exists()
    data = get_data(data_path)
    bs = batch_size if split == "train" else len(data['W'])
    loader = DataLoader(LDADataset(data['W'].type(f_type), data['Y'].type(f_type)), batch_size=bs, shuffle=True, collate_fn=collate_fn_tensor)
    return loader