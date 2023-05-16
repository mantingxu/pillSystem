import glob
import os
import shutil

test_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill2/test/*.png'
for file in glob.glob(test_path):
    # print(file)
    num = file.split('/')[-1].split('-')[0].split('_')[-1]
    name = file.split('/')[-1]
    # print(name)
    # print(num)
    src = file
    dst = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill2/test_to_train/{file_name}'.format(file_name=name)
    print(src, dst)
    if int(num) % 2 == 0:
        shutil.move(src, dst)
