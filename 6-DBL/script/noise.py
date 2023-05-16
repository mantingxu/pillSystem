import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from IPython.display import clear_output


def show_img(img):
    plt.figure(figsize=(15, 15))
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()


def gaussian_noise(img, mean=0, sigma=0.1):
    # int -> float (標準化)
    img = img / 255.0
    # 隨機生成高斯 noise (float + float)
    noise = np.random.normal(mean, sigma, img.shape)
    # noise + 原圖
    gaussian_out = img + noise
    # 所有值必須介於 0~1 之間，超過1 = 1，小於0 = 0
    gaussian_out = np.clip(gaussian_out, 0, 1)

    # 原圖: float -> int (0~1 -> 0~255)
    gaussian_out = np.uint8(gaussian_out * 255)
    # noise: float -> int (0~1 -> 0~255)
    noise = np.uint8(noise * 255)

    print("gaussian noise: ")
    show_img(noise)

    print("Picture add gaussian noise: ")
    show_img(gaussian_out)
    return gaussian_out


def img_processing(img):
    img = gaussian_noise(img)
    return img


file_name = "../dataset/test/12083/12083_0-0.png"
origin_img = cv2.imread(file_name)
print("origin picture:")
show_img(origin_img)

result_img = img_processing(origin_img)
show_img(result_img)

from numpy import expand_dims
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

# we first load the image
# image = load_img('parrot.jpg')
# we converting the image which is in PIL format into the numpy array, so that we can apply deep learning methods
# dataImage = img_to_array(result_img)
# print(dataImage)
# expanding dimension of the load image
imageNew = expand_dims(result_img, 0)
# now here below we creating the object of the data augmentation class
imageDataGen = ImageDataGenerator(rotation_range=90)
# because as we alreay load image into the memory, so we are using flow() function, to apply transformation
iterator = imageDataGen.flow(imageNew, batch_size=1)
# below we generate augmented images and plotting for visualization
for i in range(9):
    pyplot.subplot(330 + 1 + i)
    # generating images of each batch
    batch = iterator.next()
    # again we convert back to the unsigned integers value of the image for viewing
    image = batch[0].astype('uint8')
    file_name = 'output_{num}.png'.format(num=str(i))
    cv2.imwrite(file_name, image)
# we plot here raw pixel data
pyplot.imshow(image)

# visualize the the figure
pyplot.show()