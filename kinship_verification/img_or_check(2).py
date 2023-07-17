import os
import zipfile

directories = [
  '../dataset/1.Training/original' ,
  '../dataset/2.Validation/original'
]

# D:\KSH\project\kinship\data\dataset\1.Training\original\TS0001to0020


# 친가, 외가 둘 중에 한쪽에만 이미지가 있어야 함
check = True
for directory in directories :
  for current in os.listdir(directory) :
    if current[-4:] != '.zip':

      # Training 폴더의 경우 한 레벨 더 들어가야 함
      if directory == '../dataset/1.Training/original' :
        for folder in os.listdir(os.path.join(directory, current)) :
            path = os.path.join(directory, current)
            path = os.path.join(path, folder)

            # path = "../dataset/1.Training/original/TS0201to0300"

            if len(os.listdir(path)) == 1 :
              temp = os.listdir(path)
              # path = "../dataset/1.Training/original/TS0201to0300/TS001"
              path = os.path.join(path, temp[0])
              folder_a, folder_b = os.listdir(path)
            else : folder_a, folder_b = os.listdir(path)

      # Validataion 폴더의 경우 바로 접근
      else :
        # path = "../dataset/1.Validation/original/VSO801"
        path = os.path.join(directory, current)
        folder_a, folder_b = os.listdir(path)


      print(path)

      # 각각의 폴더에 존재하는 이미지 개수
      length_a = len(os.listdir(os.path.join(path, folder_a, "3.Age")))
      length_b = len(os.listdir(os.path.join(path, folder_b, "3.Age")))

      # 두 폴더에 이미지가 다 있는 경우
      if length_a > 0 and length_b > 0 :
        check = False

if check : print("dataset load success")
else : print("dataset load failed")