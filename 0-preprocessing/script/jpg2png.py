from PIL import Image
import glob
denseNet_top3_predict = ['12448', '12222', '12083']

for pill_id in denseNet_top3_predict:
    query_pic_folder = '/home/wall/finalSystem/6-DBL/dataset/train2/{pill}/*.jpg'.format(pill=pill_id)
    for query_pic_path in glob.glob(query_pic_folder):
        im1 = Image.open(query_pic_path)
        dist = query_pic_path.replace('.jpg','.png').replace('train2','train3')
        print(dist)
        im1.save(dist)