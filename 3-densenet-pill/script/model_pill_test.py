import glob
import os
from PIL import Image
from torchvision import transforms
import torch
from sklearn.metrics import confusion_matrix
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(filename="../logger/log0401-pill.txt", level=logging.INFO, format="%(message)s", filemode="w")


def get_key(l, val):
    for key, value in l.items():
        if val == value:
            return key

    return "key doesn't exist"


count = 0
top5_count = 0
error_pills = set()
trueList = []
predictList = []

test_path = '/media/wall/4TB_HDD/1211_dataset/1_test_no_aug_pill/*.png'
# test_path = '/media/wall/4TB_HDD/1211_dataset/pixel_transform/*.png'
for filename in glob.glob(test_path):
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # data preprocess
    input_tensor = preprocess(input_image)

    # unsqueeze (batch, channel, width, height)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # https://codeantenna.com/a/FMqJsklDXI
    path = "../best_weight/best_accuracy_pill.pth"
    model = torch.load(path)

    # predict model
    model.eval()

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    try:
        json_file = open('../pytorch_class_label/pill_class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)
    # with torch.no_grad():
    # output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(input_batch))
        predict = torch.softmax(output, dim=0)
        top5_prob, top5_id = torch.topk(predict, 5)
        top5_prob = top5_prob.cpu().numpy()
        top5_id = top5_id.cpu().numpy()
        # predict custom pill id (top-1 custom id)
        predict_cla = torch.argmax(predict).cpu().numpy()
        pred = class_indict[str(predict_cla)]
        # true pill ID
        truth = filename.split('/')[-1].split('_')[0]
        trueId = get_key(class_indict, truth)
        trueList.append(int(trueId))
        predictList.append(int(predict_cla))
        pre_message = 'pred: ' + pred
        truth_message = 'truth: ' + truth
        logging.info(pre_message)
        logging.info(truth_message)
        print(pred, truth)

        top5_id_numpy = top5_id
        top5_real_id_numpy = []
        for i in top5_id_numpy:
            top5_real_id_numpy.append(class_indict[str(i)])
        print(top5_real_id_numpy)
        logging.info(top5_real_id_numpy)

        if pred != truth:
            count += 1
            error_pills.add(truth)
            logging.info(filename)
        if pred not in top5_real_id_numpy:
            top5_count += 1

class_names = list(class_indict.keys())
cf_matrix = confusion_matrix(trueList, predictList)
df_cm = pd.DataFrame(cf_matrix, class_names,
                     class_names)  # https://sofiadutta.github.io/datascience-ipynbs/pytorch/Image-Classification-using-PyTorch.html

import numpy as np
import os

ndarray = np.asarray(df_cm)

os.makedirs('../csv', exist_ok=True)
df_cm.to_csv('../csv/confusion.csv')

# logging.info('confusion matrix pair')
# for i in range(len(ndarray[0])):
#     if ndarray[i][i] != 5:
#         for j in range(len(ndarray[1])):
#             if ndarray[i][j] != 0 and i != j:
#                 print(i, j)
#                 message = str(i) + ' ' + str(j) + ' ' + str(ndarray[i][j])
#                 logging.info(message)

plt.figure(figsize=(9, 6))
sns.heatmap(df_cm, annot=False, fmt="d", cmap='BuGn')
plt.xlabel("prediction")
plt.ylabel("label (ground truth)")
plt.savefig("../confusion_matrix_pic/confusion_matrix_pill_0324.png")
test_path = '/media/wall/4TB_HDD/1211_dataset/1_test_no_aug_pill/'
test_data_length = len(os.listdir(test_path))
print(test_data_length)
print(count)
print(top5_count)
accuracy = ((test_data_length - count) / test_data_length) * 100
top5_accuracy = ((test_data_length - top5_count) / test_data_length) * 100
print('test accuracy:' + str(accuracy) + '%')
print('top-5 test accuracy:' + str(top5_accuracy) + '%')
print(error_pills)
logging.info(error_pills)
