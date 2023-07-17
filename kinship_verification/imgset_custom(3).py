import os
import pandas as pd
import shutil

"""
* 압축 해제한 파일에서 3.Age 이미지들만 가지고 custom dataset을 구성함
* custom_dataset.csv 파일을 생성

- custom_dataset
    - images
      - F0001_AGE_D_18_a1.jpg
      - F0001_AGE_D_18_a2.jpg
      ...
    - custom_dataset.csv
"""


# custom_dataset/images
#               /custom_dataset.csv
# dataset (원본데이터)
# kinship_verification


# 최종 전처리 결과가 담길 폴더 생성
result_folder = "../custom_dataset"
if not os.path.exists(result_folder) :
  os.makedirs(result_folder)

# 이미지 폴더 생성
if not os.path.exists(os.path.join(result_folder, "images")) :
  os.makedirs(os.path.join(result_folder, "images"))

df = pd.DataFrame({
  'family_id' : [],
  'person_id' : [],
  'age_class' : [],
  'image_path' : [],
})

directories = [
  '../dataset/1.Training/original',
  '../dataset/2.Validation/original'
]

for directory in directories :
  for current in os.listdir(directory) :
    path = os.path.join(directory, current)
    if path[-4:] != '.zip' :

      if directory == '../dataset/1.Training/original' :
        for folder in os.listdir(os.path.join(directory, current)) :
          path = os.path.join(directory, current)
          path = os.path.join(path,folder)
          if len(os.listdir(path)) == 1 :
            temp = os.listdir(path)
            path = os.path.join(path, temp[0])
            folder_a, folder_b = os.listdir(path)
          else:
            folder_a, folder_b = os.listdir(path)

          # print(path)
          # path = ../dataset/1.Training/original\TS0001to0020\TS0001  .... ../dataset/1.Training/original\TS0001to0020\TS0020

          # 친가(A), 외가(B)에 존재하는 이미지 수
          length_a = len(os.listdir(os.path.join(path, folder_a, "3.Age")))
          length_b = len(os.listdir(os.path.join(path, folder_b, "3.Age")))

          # 친가(A) 폴더인 경우
          target = folder_a
          # 외가(A) 폴더인 경우
          if length_b > 0:
            target = folder_b

          # path = ../dataset/1.Training/original/TS0001to0020/TS0001
          full_path = os.path.join(path, target, "3.Age")
          for file in os.listdir(full_path):

            # 파일이 .jpg 혹은 .JPG 파일이 아니면 에러
            if file[-4:] != '.jpg' and file[-4:] != ".JPG":
              print(f"[Error] It's not a '.jpg' file: [{file}]")
              continue

            # F0001_AGE_D_18_a1.jpg
            # family_id : F0001, person_id : D, age_class : a1.jpg
            family_id, _, person_id, _, age_class = file.split('_')

            # a1.jpg -> a (age_class는 "a")
            age_class = age_class.split('.')[0][0]

            # print(family_id, person_id, age_class, file)
            df.loc[len(df.index)] = [family_id, person_id, age_class, file]

            # file =  F0001_AGE_D_18_a1.jpg
            # result_folder = ../custom_dataset

            # ../dataset/1.Training/original/TS0001(directory)/A(외가)(target)/3.Age/F0001_AGE_D_18_a1.jpg 를
            # ../custom_dataset/images/F0001_AGE_D_18_a1.jpg 에 복사한다
            shutil.copyfile(os.path.join(full_path, file),
                            os.path.join(result_folder, "images", file))


      elif directory == '../dataset/2.Validation/original' :
        path = os.path.join(directory, current)
        folder_a, folder_b = os.listdir(path)

        # 친가(A), 외가(B)에 존재하는 이미지 수
        length_a = len(os.listdir(os.path.join(path, folder_a, "3.Age")))
        length_b = len(os.listdir(os.path.join(path, folder_b, "3.Age")))


        # 친가(A) 폴더인 경우
        target = folder_a
        # 외가(A) 폴더인 경우
        if length_b > 0 :
          target = folder_b

        # path = ../dataset/1.Training/original/TS0001to0020/TS0001
        full_path = os.path.join(path, target, "3.Age")
        for file in os.listdir(full_path) :

          # 파일이 .jpg 혹은 .JPG 파일이 아니면 에러
          if file[-4:] != '.jpg' and file[-4:] != ".JPG" :
            print(f"[Error] It's not a '.jpg' file: [{file}]")
            continue

          # F0001_AGE_D_18_a1.jpg
          # family_id : F0001, person_id : D, age_class : a1.jpg
          family_id, _, person_id, _, age_class = file.split('_')

          # a1.jpg -> a (age_class는 "a")
          age_class = age_class.split('.')[0][0]

          # print(family_id, person_id, age_class, file)
          df.loc[len(df.index)] = [family_id, person_id, age_class, file]

          # file =  F0001_AGE_D_18_a1.jpg
          # result_folder = ../custom_dataset


          # ../dataset/1.Training/original/TS0001(directory)/A(외가)(target)/3.Age/F0001_AGE_D_18_a1.jpg 를
          # ../custom_dataset/images/F0001_AGE_D_18_a1.jpg 에 복사한다
          shutil.copyfile(os.path.join(full_path, file),
                          os.path.join(result_folder, "images", file))

# custom_dataset.csv는 13,068개의 이미지에 대한 메타 정보를 가진다.
# 속성(attribute) 목록: 'family_id', 'person_id', 'age_class', 'image_path'

# ../custom_dataset/custom_dataset.csv
df.to_csv(os.path.join(result_folder, "custom_dataset.csv"))
