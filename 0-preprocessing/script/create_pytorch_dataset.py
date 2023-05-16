import glob
import shutil
import os

# create pytorch dataset format
folders = set()
path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill2/test_to_train_pill/*.png'
for filename in glob.glob(path):
    folderName = filename.split('/')[-1].split('_')[0]
    pillName = filename.split('/')[-1]
    folders.add(folderName)
    src_path = filename
    dst_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill2/train/pill_pytorch/{pill_fold}/{pill_name}' \
        .format(pill_fold=folderName, pill_name=pillName)
    print(src_path, dst_path)
    shutil.copyfile(src_path, dst_path)

# create folder
# for folder in folders:
#     each_pill_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill/temp/all_pill_20_pytorch/{pill_fold}'.format(pill_fold=folder)
#     if not os.path.exists(each_pill_path):
#         os.makedirs(each_pill_path)
