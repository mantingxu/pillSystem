import csv
import glob
import os
import random
from sklearn.utils import shuffle

# need change var
first_pill_id = 325
second_pill_id = 4115
create_mode = 'train'

root_path = '../dataset/{create_mode}/'.format(create_mode=create_mode)
first_pill_path = sorted(glob.glob(os.path.join(root_path + str(first_pill_id), "*")))
second_pill_path = sorted(glob.glob(os.path.join(root_path + str(second_pill_id), "*")))

repeat_num = 120
csv_path = '../csv/{create_mode}.csv'.format(create_mode=create_mode)

# with open(csv_path, 'a', newline='') as csv_file:
#     writer = csv.writer(csv_file)
#     csv_content = []
#     for i in range(0, repeat_num):
#         image_path_one = random.choice(first_pill_path)
#         image_path_two = random.choice(first_pill_path)
#         csv_content.append([image_path_one, image_path_two, '0'])
#         # writer.writerow([image_path_one, image_path_two, '0'])
#
#     for i in range(0, repeat_num):
#         image_path_one = random.choice(second_pill_path)
#         image_path_two = random.choice(second_pill_path)
#         csv_content.append([image_path_one, image_path_two, '0'])
#         # writer.writerow([image_path_one, image_path_two, '0'])
#
#     for i in range(0, 2 * repeat_num):
#         image_path_one = random.choice(first_pill_path)
#         image_path_two = random.choice(second_pill_path)
#         csv_content.append([image_path_one, image_path_two, '1'])
#         # writer.writerow([image_path_one, image_path_two, '1'])
#
#
#     final_csv_content = shuffle(csv_content)
#     for row in final_csv_content:
#         writer.writerow(row)
