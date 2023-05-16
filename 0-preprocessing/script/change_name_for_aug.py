import glob
import os

folder_path = '/home/wall/Downloads/train/images/*.jpg'

count = 0
for file_path in glob.glob(folder_path):
    old_file_path = file_path
    new_file_path = file_path.split('_')[0] + '_{num}.jpg'.format(num=count)
    print(old_file_path, new_file_path)
    os.rename(old_file_path, new_file_path)
    count += 1