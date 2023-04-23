import glob
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, PILToTensor, ToPILImage, Normalize, RandomRotation, ToTensor

# 轉換公式參考 https://stackoverflow.com/questions/50474302/how-do-i-adjust-brightness-contrast-and-vibrance-with-opencv-python
contrast = 20
brightness = -20
# path = '/media/wall/4TB_HDD/1211_dataset/split_train_valid/train_without_valid/1205_2.png'
# name = '000.png'
# img = cv2.imread(path)
# path = '/media/wall/4TB_HDD/1211_dataset/split_train_valid/train_without_valid/*.png'
# path = '/media/wall/4TB_HDD/0401_new_env_all/transform/*.png'
path = '/media/wall/4TB_HDD/full_dataset/resize/*.png'


for filename in glob.glob(path):
    print(filename)
    name = filename.split('/')[-1]
    print(name)
    img = cv2.imread(filename)
    # output = img
    output = img * (contrast / 127 + 1) - contrast + brightness
        # 調整後的數值大多為浮點數，且可能會小於 0 或大於 255
        # 為了保持像素色彩區間為 0～255 的整數，所以再使用 np.clip() 和 np.uint8() 進行轉換
    output = np.clip(output, 0, 255)
    output = np.uint8(output)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(output)
    t1 = Compose([
        PILToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    my_transforms = Compose(
        [
            ToPILImage(),
            ToTensor(),
            # RandomRotation(degrees=30),
            ToPILImage()

        ]
    )
    im = my_transforms(output)
    # distPath = '/media/wall/4TB_HDD/1211_dataset/pixel_transform/' + name
    distPath = '/media/wall/4TB_HDD/full_dataset/transform/' + name
    im.save(distPath)










