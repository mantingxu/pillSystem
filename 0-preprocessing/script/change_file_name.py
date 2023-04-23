import glob
import os

folder_path = '/media/wall/4TB_HDD/full_dataset/0423_dataset/capsule/*.png'
file_path_prefix = '/media/wall/4TB_HDD/full_dataset/0423_dataset/capsule/'
for file_path in glob.glob(folder_path):
    capsule_id = file_path.split('/')[-1].replace('.png', '').split('_')[0]
    old_file_num = file_path.split('/')[-1].replace('.png', '').split('_')[-1]
    new_file_num = str(int(old_file_num) + 10)
    old_file_path = os.path.join(file_path_prefix,
                                 "{capsule_id}_{file_name}.png".format(capsule_id=capsule_id, file_name=old_file_num))
    new_file_path = os.path.join(file_path_prefix,
                                 "{capsule_id}_{file_name}.png".format(capsule_id=capsule_id, file_name=new_file_num))
    print(old_file_path, new_file_path)
    os.rename(old_file_path, new_file_path)
