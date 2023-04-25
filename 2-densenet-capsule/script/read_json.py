import glob
import json
import shutil

try:
    json_file = open('../label/capsule_class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# for id in class_indict.values():
#     print(id)
#
#     # pytorch folder
#     src_path = '/media/wall/4TB_HDD/full_dataset/pytorch_resize_sharpen/{capsule_folder}/'.format(capsule_folder=id)
#     dst_path = '/media/wall/4TB_HDD/full_dataset/capsule_dataset_sharpen/{capsule_folder}/'.format(capsule_folder=id)
#     print(src_path, dst_path)
#     shutil.move(src_path,dst_path)


for file in glob.glob('/media/wall/4TB_HDD/full_dataset/0423_dataset/compare_system_pill/*.png'):
    file_name = file.split('/')[-1]
    capsule_id = file.split('/')[-1].replace('.png', '').split('_')[0]
    if capsule_id in class_indict.values():
        # test dataset
        src_path = '/media/wall/4TB_HDD/full_dataset/0423_dataset/compare_system_pill/{file_name}'.format(
            file_name=file_name)
        dst_path = '/media/wall/4TB_HDD/full_dataset/0423_dataset/compare_system_capsule/{file_name}'.format(
            file_name=file_name)
        shutil.move(src_path, dst_path)
        print(src_path, dst_path)
