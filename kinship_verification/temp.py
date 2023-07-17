import random
import os
import pandas as pd
from shutil import copyfile

"""
fixed_val_dataset/positive 폴더에 없는 번호 폴더 추가,
fixed_test_dataset 폴더에 있는 사진을 copy 해서 넣어두고 test dataset 에서는 삭제
"""


# directory = "../fixed_val_dataset/positive"
# test_img_folder = "../fixed_test_dataset"
#
#
#
# family_set = set()
# for img in os.listdir(test_img_folder) :
#   family_id, _, person_id, _, age_class = img.split('_')
#
#   if family_id not in family_set :
#     family_set.add(family_id)
#
# family_list = list(family_set)
#
#
# for file in os.listdir(directory) :
#   # 사진이 없으면
#   if len(os.listdir(os.path.join(directory,file))) == 0 :
#
#     # family_list 에서 랜덤으로 family 하나를 뽑고
#     random_family = random.choice(family_list)
#
#     img_list = []
#     for img in os.listdir(test_img_folder):
#       family_id, _, person_id, _, age_class = img.split('_')
#       if family_id == random_family:
#         img_list.append(img)
#
#     # 그 가족에서 랜덤으로 사람 2명 사진을 뽑고
#     f1, f2 = random.sample(img_list, 2)
#
#
#     # positive 파일에 복사해주고
#     copyfile(
#       os.path.join(test_img_folder, f1),
#       os.path.join(os.path.join(directory,file), f1)
#     )
#
#     copyfile(
#       os.path.join(test_img_folder, f2),
#       os.path.join(os.path.join(directory, file), f2)
#     )
#
#
#     os.remove(os.path.join(test_img_folder, f1))
#     os.remove(os.path.join(test_img_folder, f2))





"""
fixed_test_dataset이 positive랑 negative로 분류가 안되어있어서
final_fixed_test_dataset 만들어서 분류했음
"""


# df = pd.DataFrame({
#   'family_id' : [],
#   'person_id' : [],
#   'age_class' : [],
#   'image_path' : [],
# })
#
# test_image_folder = "../fixed_test_dataset"
# for image in os.listdir(test_image_folder) :
#   family_id, _, person_id, _, age_class = image.split('_')
#   age_class = age_class.split('.')[0][0]
#   df.loc[len(df.index)] = [family_id, person_id, age_class, image]
#
# df.to_csv("../test_dataset.csv")
#
#
# def parsing(metadata) :
#
#   # 속성(attribute) 목록: 'family_id', 'person_id', 'age_class', 'image_path'
#   # 'family_id' : F0001
#   # 'person_id' : D
#   # 'age_class' : a
#   # 'image_path'  파일경로
#
#   family_set = set()
#   family_to_person_map = dict()
#   person_to_image_map = dict()
#
#   # csv 파일을 읽어서 행 읽어들임
#   for idx, row in metadata.iterrows() :
#     family_id = row['family_id']
#     person_id = row['person_id']
#     key = family_id + "_" + person_id
#     image_path = row["image_path"]
#     if family_id not in family_set :
#       family_set.add(family_id)
#       family_to_person_map[family_id] = []
#     if person_id not in family_to_person_map[family_id] :
#       # family_id (F0001)에 person_id (D, E...) 가 매핑됨
#       family_to_person_map[family_id].append(str(person_id))
#
#       person_to_image_map[key] = []
#     person_to_image_map[key].append(image_path)
#
#   family_list = list(family_set)    # family_list = ['F0001', 'F0002', ...]
#
#   return family_list, family_to_person_map, person_to_image_map
#
#
#
# image_directory = "../fixed_test_dataset"
# metadata_path = "../test_dataset.csv"
# metadata = pd.read_csv(metadata_path)
# family_list, family_to_person_map, person_to_image_map = parsing(metadata)
#
#
# # test image 총 7210개
# positive_folder = "../final_fixed_test_dataset/positive"
# if not os.path.exists(positive_folder) :
#   os.makedirs(positive_folder)
# negative_folder = "../final_fixed_test_dataset/negative"
# if not os.path.exists(negative_folder) :
#   os.makedirs(negative_folder)
#
# total_pairs = 5562
# for idx in range(total_pairs) :
#   # positive samples (family)
#   # 긍정은 짝수번째
#   if idx % 2 == 0 :
#     # 랜덤으로 한 가족을 뽑고
#     family_id = random.choice(family_list)
#     # 그 가족의 구성원 2명을 뽑음
#     p1, p2 = random.choices(family_to_person_map[family_id], k=2)
#     key1 = family_id + "_" + p1
#     key2 = family_id + "_" + p2
#     result_folder = positive_folder
#
#   # negative smaples (family)
#   # 부정은 짝수번째
#   else :
#     # 랜덤으로 두 가족을 뽑고
#     f1, f2 = random.sample(family_list, 2)
#     # 각각의 가족에서 구성원 한명씩 뽑음
#     p1 = random.choice(family_to_person_map[f1])
#     p2 = random.choice(family_to_person_map[f2])
#
#     key1 = f1 + "_" + p1
#     key2 = f2 + "_" + p2
#     result_folder = negative_folder
#
#
#   # key1 = F0001_D, key2 = F0001_GM
#   # path1 = 'F0001_AGE_D_18_a1.jpg', path2 = 'F0001_AGE_GM_18_a1.jpg'
#   path1 = random.choice(person_to_image_map[key1])
#   path2 = random.choice(person_to_image_map[key2])
#
#   if not os.path.exists(os.path.join(result_folder, str(idx // 2))) :
#     # fixed_val_dataset/positive/0
#     os.makedirs(os.path.join(result_folder, str(idx // 2)))
#
#
#   # resized_images_resolution_256/F0001_AGE_D_18_a1.jpg 를 fixed_val_dataset/positive/0/F0001_AGE_D_18_a1.jpg 에 복사한다
#   copyfile (
#     os.path.join(image_directory, path1),
#     os.path.join(result_folder, str(idx //2), path1)
#   )
#   copyfile(
#     os.path.join(image_directory, path2),
#     os.path.join(result_folder, str(idx // 2), path2)
#   )



"""
final_fixed_test_dataset 폴더에서 사진이 한장만 있는 positive 폴더가 존재함
한장만 있는 폴더를 조사해서
fixed_test_dataset 폴더에서 copy 해오기
"""

directory = "../final_fixed_test_dataset/positive"
test_dataset = '../fixed_test_dataset'

test_img = []
for img in os.listdir(test_dataset) :
  test_img.append([img.split('_')[0], img])



cnt = 0
for file in os.listdir(directory) :
  if len(os.listdir(os.path.join(directory, file))) == 1 :
    img = os.listdir(os.path.join(directory, file))[0]
    family = img.split('_')[0]
    print(family)

    temp = []
    for path in test_img :
      if path[0] == family :
        temp.append(path[1])

    for i in temp :
      if i == img :
        temp.remove(i)
    print(temp)

    random_img = random.choice(temp)
    print(random_img)
    print("\n")

    copyfile(
      os.path.join(test_dataset, random_img),
      os.path.join(directory, file, random_img)
    )
