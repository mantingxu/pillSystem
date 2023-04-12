import os
import cv2
from os import path

def draw_rectangle(img, pillSet, detect_dir, index):
    h, w, channel = img.shape
    label_txt = detect_dir + '/labels/' + str(index) + '.txt'
    with open(label_txt) as f:
        lines = f.readlines()
    # 1 pill 1 line
    for i in range(0, len(lines)):
        # pillSet key: custom pill id, value: real hospital pill id
        text = str(pillSet[i])


        line = lines[i]
        number = line.split(' ')[1:]
        # yolo to xmax xmin
        x_center, y_center, width, height = number
        x_center = float(x_center)
        y_center = float(y_center)
        width = float(width)
        height = float(height)

        xmax = int((2 * x_center * w + w * width) / 2)
        xmin = int((2 * x_center * w - w * width) / 2)
        ymax = int((2 * y_center * h + h * height) / 2)
        ymin = int((2 * y_center * h - h * height) / 2)

        left_up = (xmin, ymin)
        right_down = (xmax, ymax)
        color = (0, 0, 255)  # red
        thickness = 2  # 寬度 (-1 表示填滿)
        cv2.rectangle(img, left_up, right_down, color, thickness)
        cv2.putText(img, text, (xmin-10, ymin-10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 1, cv2.LINE_AA)
    return img


def find_same_img_pill(id, index):
    pillSet = []
    keys = list(id.keys())
    values = list(id.values())


    for i in range(0, len(keys)):
        if keys[i].split('-')[0] == str(index):
            # store custom ID (will compare with label.txt)
            pillSet.append(values[i])
    return pillSet

"""
@param pillID: Object - 
@param detect_dir: String - whole pill bag image with label
@param orignal_dir: String - whole pill bag image orignal
"""
def draw_pill_bbox(pillID, detect_dir, original_dir):
    detect_dir = str(detect_dir.as_posix())
    # export folder name
    exp_folder = detect_dir.split('/')[-1]
    # make export folder path
    os.mkdir('./runs/result/' + exp_folder)
    label_dir = detect_dir + '/labels'
    files = os.listdir(label_dir)
    imgs_name = []
    for file in files:
        imgs_name.append(file.split('.')[0])
    img_num = len(files)
    for i in range(0, img_num):
        # find_same_img_pill(predict result, predict pill real id)
        # pillID: key=> pillID(fileName), value=> customID
        pillSet = find_same_img_pill(pillID, imgs_name[i])
        img_path = original_dir + '/' + str(imgs_name[i]) + '.png'
        if not path.exists(img_path):
            continue
        # read orignal image
        img = cv2.imread(img_path)
        # draw rectangle (bounding box)
        new_img = draw_rectangle(img, pillSet, detect_dir, imgs_name[i])
        new_img_path = './runs/result/' + exp_folder + '/' + str(imgs_name[i]) + '.png'
        print(new_img_path)
        # write new image to new image path
        cv2.imwrite(new_img_path, new_img)
    return
