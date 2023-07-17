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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from facenet_pytorch import InceptionResnetV1

from itertools import combinations

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



batch_size = 64
lr = 0.005
log_step = 20
weights = [6,5,2]
weight_map = {
  'D' : 0,
  'D2' : 0,
  'D3' : 0,
  'D4' : 0,
  'S' : 0,
  'S2' : 0,
  'S3' : 0,
  'S4' : 0,
  'F' : 1,
  'M' : 1,
  'GF' : 2,
  'GM' : 2,
}


class TrainDataset(Dataset) :
  def __init__(self, meta_data, image_directory, transform=None):
    self.meta_data = meta_data
    self.image_directory = image_directory
    self.transform = transform

    family_list, family_to_person_map, person_to_image_map = parsing(meta_data)

    self.family_list = family_list
    self.family_to_person_map = family_to_person_map
    self.person_to_image_map = person_to_image_map


  def __len__(self):
    return len(self.meta_data) * 2

  def __getitem__(self, idx):
    if idx % 2 == 0:
      family_id = random.choice(self.family_list)
      p1, p2 = random.sample(self.family_to_person_map[family_id], 2)
      key1 = family_id + "_" + p1
      key2 = family_id + "_" + p2
      label = 1

    else :
      f1, f2 = random.sample(self.family_list, 2)
      p1 = random.choice(self.family_to_person_map[f1])
      p2 = random.choice(self.family_to_person_map[f2])
      key1 = f1 + "_" + p1
      key2 = f2 + "_" +p2
      label = 0

    path1 = random.choice(self.person_to_image_map[key1])
    path2 = random.choice(self.person_to_image_map[key2])

    img1 = Image.open(os.path.join(self.image_directory, path1))
    img2 = Image.open(os.path.join(self.image_directory, path2))

    # path1 = 'F0001_AGE_GM_18_a1.jpg'
    # g1 = 'GM'
    g1 = path1.split("_")[4]
    g2 = path2.split("_")[4]
    # weight_map['GM'] = 2, weight_map['F'] = 1
    # weights[1] = 5
    weight = weights[abs(weight_map[g1] - weight_map[g2])]

    if self.transform :
      img1, img2 = self.transform(img1), self.transform(img2)

    return img1, img2, label, weight




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
    if idx % 2 == 0:
      result_folder = os.path.join(self.positive_folder, self.positive_list[idx // 2])
      file1, file2 = os.listdir(result_folder)
      label = 1
    else :
      result_folder = os.path.join(self.negative_folder, self.negative_list[idx // 2])
      file1, file2 = os.listdir(result_folder)
      label = 0

    img1 = Image.open(os.path.join(result_folder, file1))
    img2 = Image.open(os.path.join(result_folder, file2))

    g1 = file1.split("_")[2]
    g2 = file2.split("_")[2]
    weight = weights[abs(weight_map[g1] - weight_map[g2])]

    if self.transform:
      img1, img2 = self.transform(img1), self.transform(img2)

    return img1, img2, label, weight





train_meta_data_path = "../custom_dataset/custom_train_dataset.csv"
train_meta_data = pd.read_csv(train_meta_data_path)

train_image_directory = "../fixed_train_dataset"
val_image_directory = "../fixed_val_dataset"
test_image_directory = "../final_fixed_test_dataset"


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





train_dataset = TrainDataset(train_meta_data, train_image_directory, train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_dataset = EvaluationDataset(val_image_directory, val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_dataset = EvaluationDataset(test_image_directory, test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)



plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 60
plt.rcParams.update({'font.size': 20})

def imshow(input) :
  input = input.numpy().transpose((1,2,0))
  mean = np.array([0.5, 0.5, 0.5])
  std = np.array([0.5, 0.5, 0.5])

  input = (std * input) + mean
  input = np.clip(input, 0, 1)
  plt.imshow(input)
  plt.show()


# iterator = iter(train_dataloader)
# img1, img2, label, weight = next(iterator)
# out = torchvision.utils.make_grid(img1[:4])
# imshow(out)
# out = torchvision.utils.make_grid(img2[:4])
# imshow(out)
# print(label[:4])


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




net = SiameseNet().cuda()
net = torch.nn.DataParallel(net)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, patience=5)
sigmoid = nn.Sigmoid()



def train() :
  start_time = time.time()
  net.train()
  total = 0
  running_loss = 0.0
  running_corrects = 0


  for i, batch in enumerate(train_dataloader) :
    optimizer.zero_grad()

    img1, img2, label, weight = batch
    img1, img2, label, weight = img1.cuda(), img2.cuda(), label.float().view(-1,1).cuda(), weight.float().view(-1,1).cuda()

    output = net(img1, img2)
    preds = sigmoid(output) >= 0.5  # 0 or 1
    loss = criterion(output, label)
    loss = loss * weight
    loss = torch.mean(loss)

    loss.backward()
    optimizer.step()

    # total은 왜 더하는거지? label이라고 해봤자 0이랑 1뿐인데...
    total += label.shape[0]

    # 현재 loss는 tensor 형태, item()을 통해 loss가 갖고 있는 스칼라값을 구함
    running_loss += loss.item()
    running_corrects += torch.sum(preds == (label >= 0.5))


    if i % log_step == log_step - 1 :
      print(f'[Batch: {i + 1}] running train loss: {running_loss / total}, running train accuracy: {running_corrects / total}')

  print(f'train loss: {running_loss / total}, accuracy: {running_corrects / total}')
  print("elapsed time:", time.time() - start_time)
  return running_loss / total, (running_corrects / total).item()



def validate() :
  start_time = time.time()
  net.eval()
  total = 0
  running_loss = 0.0
  running_corrects = 0

  for i, batch in enumerate(val_dataloader) :
    img1, img2, label, weight = batch
    img1, img2, label, weight = img1.cuda(), img2.cuda(), label.float().view(-1, 1).cuda(), weight.float().view(-1, 1).cuda()

    with torch.no_grad() :
      output = net(img1, img2)
      preds = sigmoid(output) >= 0.5
      loss = criterion(output, label)
      loss = loss * weight
      loss = torch.mean(loss)

    total += label.shape[0]
    running_loss += loss.item()
    running_corrects += torch.sum(preds == (label >= 0.5))

    if (i == 0) or (i % log_step - 1) :
      print(
        f'[Batch: {i + 1}] running val loss: {running_loss / total}, running val accuracy: {running_corrects / total}')

    print(f'val loss: {running_loss / total}, accuracy: {running_corrects / total}')
    print("elapsed time:", time.time() - start_time)
    return running_loss / total, (running_corrects / total).item()


log_folder = "../log_folder"
if not os.path.exists(log_folder) :
  os.makedirs(log_folder)

num_epochs = 20
best_val_acc = 0
best_epoch = 0

history = []
accuracy = []
for epoch in range(num_epochs) :
  train_loss, train_acc = train()
  val_loss, val_acc = validate()
  history.append((train_loss, val_loss))
  accuracy.append((train_acc, val_acc))
  scheduler.step(val_loss)

  if val_acc > best_val_acc :
    print("[Info] best validation accuracy")
    best_val_acc = val_acc
    best_epoch = epoch
    torch.save(net.state_dict(), f'{log_folder}/best_checkpoint_epoch_{epoch + 1}.pth')
  torch.save(net.state_dict(), f'{log_folder}/checkpoint_epoch_{epoch + 1}.pth')
torch.save(net.state_dict(), f'{log_folder}/last_checkpoint_epoch_{num_epochs}.pth')





plt.plot([x[0] for x in history], 'b', label='train')
plt.plot([x[1] for x in history], 'r--',label='validation')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


plt.plot([x[0] for x in accuracy], 'b', label='train')
plt.plot([x[1] for x in accuracy], 'r--',label='validation')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()


net.load_state_dict(torch.load(f'{log_folder}/best_checkpoint_epoch_' + str(best_epoch + 1) + '.pth'))






def test() :
  start_time = time.time()
  net.eval()
  total = 0
  running_loss = 0.0
  running_corrects = 0

  for i, batch in enumerate(test_dataloader) :
    img1, img2, label, weight = batch
    img1, img2, label, weight = img1.cuda(), img2.cuda(), label.float().view(-1, 1).cuda(), weight.float().view(-1, 1).cuda()

    with torch.no_grad():
      output = net(img1, img2)
      preds = sigmoid(output) >= 0.5
      loss = criterion(output, label)
      loss = loss * weight
      loss = torch.mean(loss)

    total += label.shape[0]
    running_loss += loss.item()
    running_corrects += torch.sum(preds == (label >= 0.5))

    if (i == 0) or (i % log_step == log_step - 1):
      print(
        f'[Batch: {i + 1}] running test loss: {running_loss / total}, running test accuracy: {running_corrects / total}')

  print(f'test loss: {running_loss / total}, accuracy: {running_corrects / total}')
  print("elapsed time:", time.time() - start_time)
  return running_loss / total, (running_corrects / total).item()

print("\n")
test()







checkpoints = [15, 16, 17, 18, 19]
models = []
for checkpoint in checkpoints :
  model = SiameseNet().cuda()
  model = nn.DataParallel(model)
  state_dict = torch.load(f"{log_folder}/checkpoint_epoch_{checkpoint}.pth")
  model.load_state_dict(state_dict)

  models.append(model)


def test_ensembles() :
  start_time = time.time()
  total = 0
  running_loss = 0.0
  running_corrects = 0

  for i, batch in enumerate(test_dataloader) :
    img1, img2, label, weight = batch
    img1, img2, label, weight = img1.cuda(), img2.cuda(), label.float().view(-1, 1).cuda(), weight.float().view(-1, 1).cuda()

    preds = 0
    losses = 0
    for model in models :
      model.eval()
      with torch.no_grad() :
        output = model(img1, img2)
        preds += sigmoid(output) / len(models)
        loss = criterion(output, label)
        loss = loss * weight
        loss = torch.mean(loss)
        losses += loss / len(models)

    pred = preds >= 0.5
    total += label.shape[0]
    running_loss += losses.item()
    running_corrects += torch.sum(preds == (label >= 0.5))
    if (i == 0) or (i % log_step == log_step - 1):
      print(
        f'[Batch: {i + 1}] running test loss: {running_loss / total}, running test accuracy: {running_corrects / total}')

  print(f'test loss: {running_loss / total}, accuracy: {running_corrects / total}')
  print("elapsed time:", time.time() - start_time)
  return running_loss / total, (running_corrects / total).item()

print("\n")
test_ensembles()







net = SiameseNet().cuda()
net = nn.DataParallel(net)
max = 1
for log in os.listdir("../log_folder") :
  if log.split("_")[0] == "best" :
    num = log.split("_")[-1].split(".")[0]
    if int(num) > max :
      max = int(num)
      last = log
state_dict = torch.load(f"../log_folder/{last}")
net.load_state_dict(state_dict)

print("\n")
test()





# 여기서 실제 이용 부분

images = []
image_cnt = 4

for i in range(1, image_cnt + 1):
    filename = f'image_{i}.jpg'
    image = Image.open("../example/" + filename).convert('RGB')
    image = test_transform(image).unsqueeze(0).cuda()
    images.append(image)


possibles = list(combinations(images, 2))

for possible in possibles:
    one, two = possible

    with torch.no_grad():
        output = net(one, two)
        pred = sigmoid(output) >= 0.5


    result = pred[0][0].item()
    print(f"가족 관계 예측 결과: {result}")
    out = torchvision.utils.make_grid(torch.stack((one[0], two[0])))
    imshow(out.cpu())