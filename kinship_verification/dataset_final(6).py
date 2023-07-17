import pandas as pd

"""
test 데이터셋이 조금 이상해서 마지막으로 정리
(train에는 없는데, val이랑 중복되는 이미지가 있었음)

'custom_dataset.csv' train_dataset으로만 다시 만들어주기 (원래는 전체 이미지였음)
"""



import os

test_dataset_folder = "../fixed_test_dataset"
train_dataset_folder = "../fixed_train_dataset"

result_folder = "../custom_dataset"


# list_negative = []
# list_positive = []
#
# negative_folder = "../fixed_val_dataset/negative"
# positive_folder = "../fixed_val_dataset/positive"
#
# for file in os.listdir(negative_folder) :
#   a, b = os.listdir(os.path.join(negative_folder,file))
#   list_negative.append(a)
#   list_negative.append(b)
#
# for file in os.listdir(positive_folder) :
#   if len(os.listdir(os.path.join(positive_folder,file))) == 1:
#     a = os.listdir(os.path.join(positive_folder,file))
#     os.remove(os.path.join(positive_folder,file,a[0]))
#     os.rmdir(os.path.join(positive_folder,file))
#   else :
#     a, b =  os.listdir(os.path.join(positive_folder,file))
#     list_positive.append(a)
#     list_positive.append(b)
#
#
#
# for file in os.listdir(test_dataset_folder) :
#   if file in list_positive or file in list_negative :
#     os.remove(os.path.join(test_dataset_folder, file))






# df = pd.DataFrame({
#   'family_id' : [],
#   'person_id' : [],
#   'age_class' : [],
#   'image_path' : [],
# })
#
# for file in os.listdir(train_dataset_folder) :
#   family_id, _, person_id, _, age_class = file.split('_')
#   age_class = age_class.split('.')[0][0]
#   path = os.path.join(train_dataset_folder, file)
#   df.loc[len(df.index)] = [family_id, person_id, age_class, path]
#
# df.to_csv(os.path.join(result_folder, "custom_dataset.csv"))
