import os
import cv2

"""
* 사이즈 재조정해서 저장
- resized_images_resolution_256
    - F0001_AGE_D_18_a1.jpg
    - F0001_AGE_D_18_a2.jpg
    ...
"""

# 이미지 크기 수정하고 결과가 담길 폴더 생성
result_folder = "../resized_images_resolution_256"
if not os.path.exists(result_folder) :
  os.makedirs(result_folder)

directory = "../custom_dataset/images"
for file in os.listdir(directory) :
  # 이미지 읽어
  path = os.path.join(directory, file)
  img = cv2.imread(path)

  # 256 x 256 사이즈로 변환
  resized_img = cv2.resize(img, (256,256))

  saved_path = os.path.join(result_folder, file)
  cv2.imwrite(saved_path, resized_img)