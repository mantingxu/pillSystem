import cv2
import glob
import os

IMAGE_SIZE = 200


def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)

    # 获取图像尺寸
    h, w, _ = image.shape

    # 对于长宽不相等的图片，找到最长的一边
    # longest_edge = max(h, w)
    longest_edge = IMAGE_SIZE

    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    if w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    # print(top, bottom, left, right)
    # RGB颜色
    BLACK = [0, 0, 0]

    # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    # 调整图像大小并返回
    return cv2.resize(constant, (height, width))


def split_capsule_pill(folder_name, resize_path, detect_dir):
    split_capsule_path = 'runs/split/' + folder_name + '/capsule.txt'
    split_pill_path = 'runs/split/' + folder_name + '/pill.txt'
    split_folder_path = 'runs/split/' + folder_name
    if not os.path.exists(split_folder_path):
        os.makedirs(split_folder_path)
    for resize_img_path in sorted(glob.glob(resize_path + '/*.png')):
        # print(resize_img_path)
        line_number = resize_img_path.split('/')[-1].split('.')[0].split('-')[-1].replace('.png', '')
        if int(line_number) > 0:
            print(resize_img_path)

        name = resize_img_path.split('/')[-1].split('.')[0].split('-')[0]
        # print(name)
        label_path = detect_dir + '/labels/' + str(name) + '.txt'
        # print(label_path)
        if not os.path.exists(label_path):
            # print('==========================================')
            # print(str(name))
            continue
        capsule_message = []
        pill_message = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
        # for line in lines:
        yolo_arr = lines[int(line_number)].split(' ')
        class_name, x_center, y_center, width, height = yolo_arr
        if class_name == '0':
            pill_message.append(resize_img_path + '\n')
        else:
            capsule_message.append(resize_img_path + '\n')
        if os.path.exists(split_capsule_path):
            capsule_append_write = 'a'  # append if already exists
        else:
            capsule_append_write = 'w'  # make a new file if not
        if os.path.exists(split_pill_path):
            pill_append_write = 'a'  # append if already exists
        else:
            pill_append_write = 'w'  # make a new file if not
        with open(split_capsule_path, capsule_append_write) as f:
            for message in capsule_message:
                f.write(message)
        with open(split_pill_path, pill_append_write) as f:
            for message in pill_message:
                f.write(message)
    return [split_capsule_path, split_capsule_path]


def resize_pill(crop_dir, detect_dir):
    if not isinstance(detect_dir, str):
        detect_dir = str(detect_dir.as_posix())
    folder_name = crop_dir.split('/')[-1]  # expN
    resize_path = 'runs/resize/' + folder_name
    if not os.path.isdir(resize_path):
        os.makedirs(resize_path)
    imgs = []
    for i in sorted(glob.glob(crop_dir + '/*.png')):
        name = i.split('/')[-1].split('.')[0]
        imgs.append(name)  # store image name
        path = crop_dir + '/' + str(name) + '.png'
        img = cv2.imread(path)
        res = resize_image(img)
        res_path = resize_path + '/' + name + '.png'
        cv2.imwrite(res_path, res)
    [capsule_class_txt, pill_class_txt] = split_capsule_pill(folder_name, resize_path, detect_dir)
    return [resize_path, capsule_class_txt, pill_class_txt]
