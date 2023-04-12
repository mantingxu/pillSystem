import os
from datetime import date


def folder_name():
    today = date.today()
    current_date = today.strftime("%Y%m%d")
    folder_path = "./temp-dataset/dataset" + current_date + "/"
    # 建立第一個資料夾 (以日期為主)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    all_dir = os.listdir(folder_path)
    for i in range(0, len(all_dir)):
        all_dir[i] = int(all_dir[i])
    if len(all_dir) == 0:
        max_in_dir = 0
    else:
        max_in_dir = max(all_dir)
    # create second layer folder(folder name will increment)
    path = './temp-dataset/dataset' + current_date + '/' + str(max_in_dir + 1)
    os.makedirs(path)
    return path







