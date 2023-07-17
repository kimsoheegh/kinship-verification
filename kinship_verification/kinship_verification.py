import random
import os
import time
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from facenet_pytorch import InceptionResnetV1
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import torchvision

import time




def parsing(metadata) :
  family_set = set()
  family_to_person_map = dict()
  person_to_image_map = dict()

  for idx, row in metadata.iterrows() :
    family_id = row["family_id"]
    person_id = row["person_id"]
    key = family_id + "_" + person_id
    image_path = row["image_path"]

    if family_id not in family_set :
      family_set.add(family_id)
      family_to_person_map[family_id] = []

    if person_id not in family_to_person_map[family_id] :
      family_to_person_map[family_id].append(str(person_id))
      person_to_image_map[key] = []
    person_to_image_map[key].append(image_path)

  family_list = list(family_set)
  return family_list, family_to_person_map, person_to_image_map


# 학습 데이터셋
class TrainDataset(Dataset) :
  def __init__(self, meta_data, image_directory, transform=None):
    self.meta_data = meta_data
    self.image_directory = image_directory
    self.transform = transform

    # family_list = ['F0001', 'F0002', ...]
    # family_to_person_map = { 'F0001' : ['D', 'E', ...], 'F0002' : ['D', 'E', ...] }
    # person_to_image_map = { 'F0001_D' : 'F0001_AGE_D_18_a1.jpg' }

    family_list, family_to_person_map, person_to_image_map = parsing(meta_data)

    self.family_list = family_list
    self.family_to_person_map = family_to_person_map
    self.person_to_image_map = person_to_image_map

  def __len__(self):
    return len(self.meta_data) * 2

  def __getitem__(self, idx):
    # label = positive pair = 1
    if idx % 2 == 0 :
      family_id = random.choice(self.family_list)
      # 여기 안되면 random.choices로 바꿔보기
      p1, p2 = random.sample(self.family_to_person_map[family_id], 2)
      key1 = family_id + "_" + p1
      key2 = family_id + "_" + p2
      label = 1

    # label = negative pair = 0
    else :
      f1, f2 = random.sample(self.family_list, 2)
      p1 = random.choice(self.family_to_person_map[f1])
      p2 = random.choice(self.family_to_person_map[f2])
      key1 = f1 + "_" + p1
      key2 = f2 + "_" + p2
      label = 0

    path1 = random.choice(self.person_to_image_map[key1])
    path2 = random.choice(self.person_to_image_map[key2])

    img1 = Image.open(os.path.join(self.image_directory, path1))
    img2 = Image.open(os.path.join(self.image_directory, path2))

    if self.transform :
      img1, img2 = self.transform(img1), self.transform(img2)

    return img1, img2, label



# 평가 데이터셋
class EvaluationDataset(Dataset) :
  def __init__(self, image_directory, transform=None):
    self.positive_folder = os.path.join(image_directory, "positive")
    self.negative_folder = os.path.join(image_directory, "negative")
    self.positive_list = os.listdir(self.positive_folder)
    self.negative_list = os.listdir(self.negative_folder)
    self.transform = transform

  def __len__(self):
    return len(self.positive_list) + len(self.negative_list)

  def __getitem__(self, idx):
    # positive (가족)
    if idx % 2 == 0 :
      result_folder = os.path.join(self.positive_folder, self.positive_list[idx // 2])  # fixed_val_dataset/positive/0
      file1, file2 = os.listdir(result_folder)                                          # F0001_AGE_D_18_a1.jpg, F0001_AGE_D_18_a2.jpg
      label = 1

    # negative (가족 아님)
    else :
      result_folder = os.path.join(self.negative_folder, self.negative_list[idx // 2])
      file1, file2 = os.listdir(result_folder)

      label = 0

    img1 = Image.open(os.path.join(result_folder, file1))                              # fixed_val_dataset/positive/0/F0001_AGE_D_18_a2.jpg
    img2 = Image.open(os.path.join(result_folder, file2))

    if self.transform :
      img1, img2 = self.transform(img1), self.transform(img2)

    return img1, img2, label




# 학습 데이터셋
train_meta_data_path = "../custom_dataset/custom_train_dataset.csv"
train_meta_data = pd.read_csv(train_meta_data_path)

train_image_directory = "../fixed_train_dataset"
val_image_directory = "../fixed_val_dataset"
test_image_directory = "../fixed_test_dataset"



train_transform = transforms.Compose([
  transforms.Resize(256),

  # 이미지를 랜덤으로 좌우반전
  transforms.RandomHorizontalFlip(),

  # 이미지를 Channel x Height x Width 로 (HWC를 CHW로 바꿈), 0~255에서 0~1 로 바꿔줌
  transforms.ToTensor(),

  # (픽셀 값 - 평균(mean)) / 표준편차(std)
  # 각각의 평균, 표준편차를 산출, 각 RGB 값을 통일해 정규화 진행 (네트워크가 데이터 처리 쉬워짐)
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])



# train_meta_data = pd.read_csv("../custom_dataset/custom_train_dataset.csv")
# train_image_directory = "../fixed_train_dataset"
train_dataset = TrainDataset(train_meta_data, train_image_directory, train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

val_dataset = EvaluationDataset(val_image_directory, val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=0)

# test_dataset = EvaluationDataset(test_image_directory, test_transform)
# test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=8)


plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 60
plt.rcParams.update({'font.size' : 20})

def imshow(input) :
  input = input.numpy().transpose((1, 2, 0))  # input default : numpy
  # CHW를 HWC로, 채널 순서를 바꿔줌

  mean = np.array([0.5, 0.5, 0.5])
  std = np.array([0.5, 0.5, 0.5])

  input = (std * input) + mean
  # max보다 큰 값을 max로 바꿔주는 함수 clip
  input = np.clip(input, 0, 1)
  plt.imshow(input)
  plt.show()

# train_dataloader 에는 img1, img2, label 이 세개가 return 되는데, next가 걔를 하나씩 꺼내줌
iterator = iter(train_dataloader)
img1, img2, label = next(iterator)
out = torchvision.utils.make_grid(img1[:4])
imshow(out)
out = torchvision.utils.make_grid(img2[:4])
imshow(out)
print(label[:4]) # 1 : family, 0 : not family


class SiameseNet(nn.Module) :
  def __init__(self):
    super().__init__()

    # 512차원의 임베딩을 반환
    # face-net 모듈에 해당 모델과 MTCNN 모델 두가지가 있으나 우리는 이걸 사용
    # (근데 MTCNN 사용하는게 더 좋다고 함)
    self.encoder = InceptionResnetV1(pretrained = 'vggface2')
    self.emb_dim = 512

    self.last = nn.Sequential(
      nn.Linear(4 * self.emb_dim, 256), # input tensor = 4 * 512, output tensor = 256
      nn.BatchNorm1d(256),              # 256 채널 입력에서 정규화
      nn.ReLU(),                        # 활성화 함수 ReLU
      nn. Linear(256, 1)                # input tensor = 256, output tensor = 1
    )

  def forward(self, input1, input2):

    # 이미지1, 이미지2
    # 두 이미지에서 얼굴을 인식함, 반환형은 텐서
    emb1 = self.encoder(input1)
    emb2 = self.encoder(input2)

    # pow(a,b) = a의 b승을 텐서로 반환 (a^b)
    x1 = torch.pow(emb1, 2) - torch.pow(emb2, 2)
    x2 = torch.pow(emb1 - emb2, 2)
    x3 = emb1 * emb2
    x4 = emb1 + emb2

    # 텐서를 1차원으로 합쳐주기
    x = torch.cat((x1, x2, x3, x4), dim = 1)

    x = self.last(x)
    return x


learning_rate = 0.001
log_step = 20

# 모델 네트워크 정의
net = SiameseNet().cuda()
net = torch.nn.DataParallel(net)  # 데이터를 임의로 분할에서 여러 GPU로 돌리게 해줌


criterion = nn.BCEWithLogitsLoss()                                # 이진 분류에서 쓰는 비용함수 (0-negative, 1-positive)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 지역 최소값 탈출
scheduler = ReduceLROnPlateau(optimizer, patience=10)             # 에러값 더 안내려갈때 learning rate를 더 감소시켜줌
sigmoid = nn.Sigmoid()                                            # 활성화 함수

num_epochs = 10

def train() :
  start_time = time.time()
  net.train()   # train 모드로 바꿔준다고 단순히 알고있으면 될듯
  total = 0
  running_loss = 0.0
  running_corrects = 0


  # batch 에서 TrainDataset에서 return 값으로 img1, img2, label, weight 들어있음
  for i, batch in enumerate(train_dataloader) :
    optimizer.zero_grad()

    img1, img2, label = batch
    img1, img2, label = img1.cuda(), img2.cuda(), label.float().view(-1,1).cuda()

    # 샴 네트워크 통과
    output = net(img1, img2)
    
    # 출력층 활성화 함수
    preds = sigmoid(output) >= 0.5

    loss = criterion(output, label)
    loss.backward()
    optimizer.step()

    total += label.shape[0]
    running_loss += loss.item()
    running_corrects += torch.sum(preds == (label >= 0.5))

    # log_step = 20, i % 20 == 19, i는 19, 39, 59, 79 ... , i+1는 20, 40, 60, 80...
    if i % log_step == log_step - 1 :
      print(f'[Batch: {i + 1}] running train loss: {running_loss / total}, running train accuracy: {running_corrects / total}')

    print(f'train loss: {running_loss / total}, accuracy: {running_corrects / total}')
    print("elapsed time:", time.time() - start_time)
    return running_loss / total, running_corrects / total


def validate() :
  start_time = time.time()
  net.eval()  # 이제 여기서는 eval 모드
  total = 0
  running_loss = 0.0
  running_corrects = 0

  for i, batch in enumerate(val_dataloader) :
    img1, img2, label = batch
    img1, img2, label = img1.cuda(), img2.cuda(), label.float().view(-1, 1).cuda()

    with torch.no_grad() :
      output = net(img1, img2)
      preds = sigmoid(output) >= 0.5    # sigmoid output은 tensor
      loss = criterion(output, label)

    total += label.shape[0]
    running_loss += loss.item()
    running_corrects += torch.sum(preds == (label >= 0.5))

    # if (i == 0) or (i % log_step == log_step - 1) :
      # print(f'[Batch: {i+1}] running val loss: {running_loss / total}, running val accuracy: {running_corrects / total}')

    # print(f'val loss: {running_loss / total}, accuracy: {running_corrects / total}')
    # print("elapsed time:", time.time() - start_time)
    return running_loss / total, running_corrects / total


num_epochs = 10
best_val_loss = 1e9
best_epoch = 0

history = []
accuracy = []
for epoch in range(num_epochs):
    train_loss, train_acc = train()
    val_loss, val_acc = validate()
    history.append((train_loss, val_loss))
    accuracy.append((train_acc, val_acc))
    scheduler.step(val_loss)


    if val_loss < best_val_loss:
        print("[Info] best validation accuracy!")
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(net.state_dict(), f'best_checkpoint_epoch_{epoch + 1}.pth')

torch.save(net.state_dict(), f'last_checkpoint_epoch_{num_epochs}.pth')


plt.plot([x[0] for x in history], 'b', label='train')
plt.plot([x[1] for x in history], 'r--',label='validation')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()