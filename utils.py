
import numpy as np, os
from torch.utils.data import  DataLoader, Dataset
import torchvision.datasets as datasets
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from jax import numpy as jnp
from sklearn.metrics import f1_score

import torch
import h5py

class parameters:

  seed = 24


  model = 'mlp'
  lr = 1e-2
  momentum = 0.9
  batch_size = 128
  epoches = 20
  opt = 'sgd'
  decay = 1e-6


def imshow(img):

    img = img * 78.567 + 33.31
    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()

def plot_training(history, path):
    
    plt.plot(history['error'])
    plt.plot(history['dev_error'])
    plt.legend(['error', 'dev_error'], loc='upper left')
    plt.ylabel('Error')
    plt.xlabel('Epoch')

    plt.savefig(os.path.join(path, 'train_history.png'))

def load_dataset(batch_size):

  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize(0, 255),
      transforms.Lambda(lambda x: x.reshape(-1))])

  train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform),
                                            batch_size=batch_size,
                                            shuffle=True)

  dev_loader = DataLoader(datasets.MNIST(root='./data', train=False, download=True, transform=transform),
                                          batch_size=batch_size,
                                          shuffle=False)
  
  return train_loader, dev_loader

def compute_macro_f1(logits, labels):

  logits = jnp.argmax(logits, -1)
  return f1_score(y_true=labels, y_pred = logits, average='macro')

def compute_accuracy(logits, labels):

  logits = jnp.argmax(logits, -1)
  return jnp.mean(logits == labels)
