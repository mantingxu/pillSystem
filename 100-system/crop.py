import cv2
import glob
import os


def crop_pill(detect_dir, original_dir):
    if not isinstance(detect_dir, str):
        detect_dir = str(detect_dir.as_posix())
    source_dir = str(detect_dir).split('/')[-1]
    crop_path = './runs/crop/' + source_dir
    if not os.path.isdir(crop_path):
        os.makedirs(crop_path)

    imgs = []
    for img in glob.glob(detect_dir + '/' + '*.png'):
        img_name = (img.split('/')[-1].split('.')[0])
        imgs.append(img_name)
    imgs = sorted(imgs)

    for name in imgs:
        path = original_dir + '/' + str(name) + '.png'
        # read original image
        img = cv2.imread(path)
        h, w, channel = img.shape
        # read label
        label_path = detect_dir + '/labels/' + str(name) + '.txt'

        if not os.path.exists(label_path):
            # print('==========================================')
            # print(str(name))
            continue
        with open(label_path, 'r') as f:
            line = f.readlines()
        count = 0  # count 1 image has how many pill
        for i in line:
            s = i.split(' ')
            class_name, x_center, y_center, width, height = s
            x_center = float(x_center)
            y_center = float(y_center)
            width = float(width)
            height = float(height)

            xmax = int((2 * x_center * w + w * width) / 2)
            xmin = int((2 * x_center * w - w * width) / 2)
            ymax = int((2 * y_center * h + h * height) / 2)
            ymin = int((2 * y_center * h - h * height) / 2)
            w0 = xmax - xmin
            h0 = ymax - ymin

            crop_img = img[ymin:ymin + h0, xmin:xmin + w0]
            file_name = crop_path + '/' + name + '-' + str(count) + '.png'
            count += 1
            if count > 1:
                print(file_name)
            cv2.imwrite(file_name, crop_img)
    return crop_path
