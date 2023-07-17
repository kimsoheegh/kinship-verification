


from PIL import Image
from torchvision import transforms

test_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


images = []
image_cnt = 4

for i in range(1, image_cnt + 1):
    filename = f'image_{i}.jpg'
    image = Image.open("../example/" + filename).convert('RGB')
    image.show()
    image = test_transform(image).unsqueeze(0).cuda()
    images.append(image)