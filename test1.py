import torch
import tqdm
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
def load_mnist(batch_size=128, path='./data'):
    train_loader = DataLoader(datasets.MNIST(path,
                                             train=True,
                                             download=True,
                                             transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
                                             ), batch_size=batch_size, shuffle=True
                              )
    test_loader = DataLoader(datasets.MNIST(path,
                                             train=False,
                                             download=True,
                                             transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
                                             ), batch_size=batch_size, shuffle=True
                              )
    return train_loader, test_loader
train_loader, test_loader = load_mnist()
print("finish")
