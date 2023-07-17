import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms



class SiameseNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.encoder = InceptionResnetV1(pretrained='vggface2')
    self.emb_dim = 512

    self.last = nn.Sequential(
      nn.Linear(4 * self.emb_dim, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(),
      nn.Linear(256, 1)
    )

  def forward(self, input1, input2):
    emb1 = self.encoder(input1)
    emb2 = self.encoder(input2)

    x1 = torch.pow(emb1, 2) - torch.pow(emb2, 2)
    x2 = torch.pow(emb1 - emb2, 2)
    x3 = emb1 * emb2
    x4 = emb1 + emb2

    x = torch.cat((x1, x2, x3, x4), dim=1)
    x = self.last(x)

    return x


def imshow(input) :
  input = input.numpy().transpose((1,2,0))
  mean = np.array([0.5, 0.5, 0.5])
  std = np.array([0.5, 0.5, 0.5])

  input = (std * input) + mean
  input = np.clip(input, 0, 1)
  plt.imshow(input)
  plt.show()


test_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


sigmoid = nn.Sigmoid()
