import glob
import json
import shutil

try:
    json_file = open('../resource/pill_class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

for id in class_indict.values():
    # print(id)

    # pytorch folder
    src_glob = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill2/test_to_train/{capsule_folder}_*'.format(capsule_folder=id)
    for file in glob.glob(src_glob):
        fileName = file.split('/')[-1]
        dst_glob = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill2/capsule/{name}'.format(name=fileName)
        print(file,dst_glob)
        shutil.move(file,dst_glob)



# for file in glob.glob('/media/wall/4TB_HDD/full_dataset/0423_dataset/compare_system_pill/*.png'):
#     file_name = file.split('/')[-1]
#     capsule_id = file.split('/')[-1].replace('.png', '').split('_')[0]
#     if capsule_id in class_indict.values():
#         # test dataset
#         src_path = '/media/wall/4TB_HDD/full_dataset/0423_dataset/compare_system_pill/{file_name}'.format(
#             file_name=file_name)
#         dst_path = '/media/wall/4TB_HDD/full_dataset/0423_dataset/compare_system_capsule/{file_name}'.format(
#             file_name=file_name)
#         shutil.move(src_path, dst_path)
#         print(src_path, dst_path)
