import pandas as pd
import random
import os
from shutil import copyfile

def parsing(metadata) :

  # 속성(attribute) 목록: 'family_id', 'person_id', 'age_class', 'image_path'
  # 'family_id' : F0001
  # 'person_id' : D
  # 'age_class' : a
  # 'image_path'  파일경로

  family_set = set()
  family_to_person_map = dict()
  person_to_image_map = dict()

  # csv 파일을 읽어서 행 읽어들임
  for idx, row in metadata.iterrows() :
    family_id = row['family_id']
    person_id = row['person_id']
    key = family_id + "_" + person_id
    image_path = row["image_path"]
    if family_id not in family_set :
      family_set.add(family_id)
      family_to_person_map[family_id] = []
    if person_id not in family_to_person_map[family_id] :
      # family_id (F0001)에 person_id (D, E...) 가 매핑됨
      family_to_person_map[family_id].append(str(person_id))

      person_to_image_map[key] = []
    person_to_image_map[key].append(image_path)

  family_list = list(family_set)    # family_list = ['F0001', 'F0002', ...]

  return family_list, family_to_person_map, person_to_image_map



metadata_path = "../custom_dataset/custom_dataset.csv"
metadata = pd.read_csv(metadata_path)
image_directory = "../resized_images_resolution_256"
family_list, family_to_person_map, person_to_image_map = parsing(metadata)

# 평가 데이터 세트가 담길 폴더
positive_folder = "../fixed_val_dataset/positive"
if not os.path.exists(positive_folder) :
  os.makedirs(positive_folder)
negative_folder = "../fixed_val_dataset/negative"
if not os.path.exists(negative_folder) :
  os.makedirs(negative_folder)

# family_list = ['F0001', 'F0002', ...]
# family_to_person_map = { 'F0001' : ['D', 'E', ...], 'F0002' : ['D', 'E', ...] }
# person_to_image_map = { 'F0001_D' : 'F0001_AGE_D_18_a1.jpg' }



total_pairs = 5562
for idx in range(total_pairs) :
  # positive samples (family)
  # 긍정은 짝수번째
  if idx % 2 == 0 :
    # 랜덤으로 한 가족을 뽑고
    family_id = random.choice(family_list)
    # 그 가족의 구성원 2명을 뽑음
    p1, p2 = random.choices(family_to_person_map[family_id], k=2)
    key1 = family_id + "_" + p1
    key2 = family_id + "_" + p2
    result_folder = positive_folder

  # negative smaples (family)
  # 부정은 짝수번째
  else :
    # 랜덤으로 두 가족을 뽑고
    f1, f2 = random.sample(family_list, 2)
    # 각각의 가족에서 구성원 한명씩 뽑음
    p1 = random.choice(family_to_person_map[f1])
    p2 = random.choice(family_to_person_map[f2])

    key1 = f1 + "_" + p1
    key2 = f2 + "_" + p2
    result_folder = negative_folder


  # key1 = F0001_D, key2 = F0001_GM
  # path1 = 'F0001_AGE_D_18_a1.jpg', path2 = 'F0001_AGE_GM_18_a1.jpg'
  path1 = random.choice(person_to_image_map[key1])
  path2 = random.choice(person_to_image_map[key2])

  if not os.path.exists(os.path.join(result_folder, str(idx // 2))) :
    # fixed_val_dataset/positive/0
    os.makedirs(os.path.join(result_folder, str(idx // 2)))


  # resized_images_resolution_256/F0001_AGE_D_18_a1.jpg 를 fixed_val_dataset/positive/0/F0001_AGE_D_18_a1.jpg 에 복사한다
  copyfile (
    os.path.join(image_directory, path1),
    os.path.join(result_folder, str(idx //2), path1)
  )
  copyfile(
    os.path.join(image_directory, path2),
    os.path.join(result_folder, str(idx // 2), path2)
  )

  # 예를 들어, fixed_val_dataset/positive/0 폴더에 뽑았던 긍정 쌍 가족 구성원 두 명의 이미지가 들어간다


# resized_images_resolution_256 파일에서 나머지 train, test 데이터셋 만들기 주기
# 16685(train) + 5562(validation) + 5562(test)

train_dataset_folder = "../fixed_train_dataset"
if not os.path.exists(train_dataset_folder) :
  os.makedirs(train_dataset_folder)

test_dataset_folder = "../fixed_test_dataset"
if not os.path.exists(test_dataset_folder) :
  os.makedirs(test_dataset_folder)



# 256 폴더에서 fixed_val_dataset 이미지 지우기
val_dir = ["../fixed_val_dataset/positive",
           "../fixed_val_dataset/negative"]

for dir in val_dir :
  print(dir)
  for pair in os.listdir(dir) :
    print(pair)
    for file in os.listdir(os.path.join(dir, pair)) :
      print(os.path.join(dir, pair))
      print(file)
      if file in image_directory :
        os.remove(os.path.join(image_directory, file))


# 학습 데이터셋 폴더
i = 0
for file in os.listdir(image_directory) :
  if i == 16685 : break
  copyfile(
    os.path.join(image_directory, file),
    os.path.join(train_dataset_folder, file)
  )
  os.remove(os.path.join(image_directory, file))
  i += 1



# 테스트 데이터셋 폴더
for file in os.listdir(image_directory) :
  copyfile(
    os.path.join(image_directory, file),
    os.path.join(test_dataset_folder, file)
  )




"""
* 최종적으로 다음과 같은 형태로 저장된다.
- kinship_verification
- dataset
- resized_images_resolution_256
    - F0001_AGE_D_18_a1.jpg 
    - F0001_AGE_D_18_a2.jpg 
    ...
- custom_dataset
    - images
        - F0001_AGE_D_18_a1.jpg 
        - F0001_AGE_D_18_a2.jpg 
        ...
    - custom_dataset.csv
- fixed_train_dataset
_ fixed_test_dataset
- fixed_val_dataset
    - positive
        - 0
          - F0001_AGE_D_18_a1.jpg 
          - F0001_AGE_E_18_a2.jpg 
        - 1
          - F0034_AGE_E_18_d1.jpg 
          - F0034_AGE_E_18_d2.jpg 
        ...
    - negative
        - 0
          - F0001_AGE_D_18_a1.jpg 
          - F00892_AGE_E_18_d1.jpg
        - 1
          - F0603_AGE_E_18_d1.jpg 
          - F0052_AGE_D_18_a1.jpg 
        ...
"""