import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import cv2

pic_one = '../dataset/test/12222/12222_20-0.png'


img_cv = cv2.imread(pic_one)
img_resize_cv = cv2.resize(img_cv, (224,224))


img = Image.open(pic_one)
transform = T.Resize(224)
img_resize = transform(img)

cv2.imwrite('output.jpg', img_cv)
cv2.imwrite('output1.jpg', img_resize_cv)
# plt.imshow(img)
# print("Size after resize:", img.size)
# plt.show()

