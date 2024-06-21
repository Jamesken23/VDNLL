
import torch, random
import numpy as np
from torch.utils.data import Dataset

NO_LABEL = -1

class DataSetWarpper(Dataset):
    """Enable dataset to output index of sample
    """
    def __init__(self, datasets, num_classes):
        self.datasets = datasets
        self.num_classes = num_classes

    def __getitem__(self, index):
        sample, label = self.datasets[index]
        return sample, label, index

    def __len__(self):
        return len(self.datasets)



class TransformWeakTwice(Dataset):
    """
    generate two input, weak sample和strong sample
    """
    def __init__(self, datasets, num_classes, vocab2id, device):
        self.datasets = datasets
        self.num_classes = num_classes
        self.vocab2id = vocab2id

        self.device = device

    def __getitem__(self, index):
        sample, label = self.datasets[index]
        weak_sample_1 = get_weak_data(sample, 20, self.vocab2id, self.device)
        weak_sample_2 = get_weak_data(sample, 20, self.vocab2id, self.device)

        return sample, weak_sample_1, weak_sample_2, label

    def __len__(self):
        return len(self.datasets)


def get_weak_data(sample, k, vocab2id, device):
    """
    同义词替换
    """
    if device:
        sample_numpy = sample.cpu().detach().numpy()
    else:
        sample_numpy = sample.detach().numpy()

    for i in range(k):
        idx = random.randint(0, len(sample_numpy)-1)
        repalce_char = random.randint(0, len(vocab2id)-1)
        sample_numpy[idx] = repalce_char

    # print("We have got weak augement data, we get {0} weak data at randomly!".format(k))
    return torch.LongTensor(sample_numpy)