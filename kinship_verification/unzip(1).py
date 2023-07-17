import os
import zipfile

directories = [
  '../dataset/1.Training/labeled',
  '../dataset/1.Training/original',
  '../dataset/2.Validation/labeled',
  '../dataset/2.Validation/original',
]


for directory in directories :
  for current in os.listdir(directory) :
    # 압축 파일까지 합한 전체 경로
    path = os.path.join(directory, current)
    # path = "./dataset/1.Training/original/TS0001to0020.zip"
    if path[-4:] == '.zip' : # 압축 파일인 경우
      compressed = zipfile.ZipFile(path)
      compressed.extractall(path[:-4])
      compressed.close()