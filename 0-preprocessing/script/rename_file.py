import glob
import os
src_glob = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill2/test_capsule/*.png'
for file in glob.glob(src_glob):
    # print(file)
    res = file.replace('.png','',1)
    print(res)
    # fileName = file.split('/')[-1]
    # dst_glob = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill2/test_capsule/{name}'.format(name=fileName)
    # print(src,dst)
    os.rename(file,res)