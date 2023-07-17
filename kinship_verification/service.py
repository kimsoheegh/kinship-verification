from PIL import Image
import torchvision
import torch
from itertools import combinations
import torch.nn as nn
from service_SiameseNet import SiameseNet
from service_SiameseNet import sigmoid
from service_SiameseNet import test_transform
from service_SiameseNet import imshow
import os

net = SiameseNet().cuda()
net = nn.DataParallel(net)

max = 1
for log in os.listdir("../log_folder") :
  if log.split("_")[0] == "best" :
    num = log.split("_")[-1].split(".")[0]
    if int(num) > max :
      max = int(num)
      last = log


# 모델 불러오기
state_dict = torch.load(f"../log_folder/{last}")
net.load_state_dict(state_dict)


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
        net.eval()
        output = net(one, two)
        pred = sigmoid(output) >= 0.5


    result = pred[0][0].item()
    print(f"가족 관계 예측 결과: {result}")
    out = torchvision.utils.make_grid(torch.stack((one[0], two[0])))
    imshow(out.cpu())