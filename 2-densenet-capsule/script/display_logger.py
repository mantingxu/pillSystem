import cv2
import numpy as np
import matplotlib.pyplot as plt

path = '../logger/logger-error_append.txt'
line_count = 0


def show_images(urls):
    print(urls)
    w = 10
    h = 10
    columns = 6
    rows = 1
    fig = plt.figure(figsize=(6, 3))
    for i in range(1, columns * rows + 1):
        capsule_id = urls[i - 1].split('/')[-1].split('_')[0]
        img = cv2.imread(urls[i - 1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig.add_subplot(rows, columns, i)
        ax = plt.subplot(160+i)
        ax.set_title(capsule_id)
        plt.imshow(img)
    plt.show()


with open(path) as f:
    display_urls = []
    for line in f.readlines():
        if line_count % 4 == 2:
            display_urls = []
            # img = cv2.imread(line)
            display_urls.append(line.replace('\n',''))
        if line_count % 4 == 3:
            top3_list = line.replace("'", '').replace('[', '').replace(']', '').replace(' ', '').replace('\n', '').split(',')
            for el in top3_list:
                print(el)
                url = '/media/wall/4TB_HDD/full_dataset/0423_dataset/test_capsule_sharpen/{pill_id}_0-0.png'.format(
                    pill_id=el)
                display_urls.append(url)
            print(display_urls)
            show_images(display_urls)
        line_count += 1
        # show_images(display_urls)
