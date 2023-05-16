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
import numpy as np
import torch.nn as nn
count = 1


def get_key(l, val):
    for key, value in l.items():
        if val == value:
            return key
    return -1


def predict_capsule_id():
    logger_path = '../logger/logger-error_append_4.txt'
    logging.basicConfig(filename=logger_path, level=logging.INFO, format="%(message)s", filemode="w")
    print('logger path: ', logger_path)
    count = 0
    top5_count = 0
    error_pills = set()
    trueList = []
    predictList = []

    test_path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill2/test/*.png'

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
        path = "../weight/best_accuracy_pill_0517.pth"
        model = torch.load(path)
        # predict model
        model.eval()

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        try:
            json_file = open('../label/new_pill_class_indices.json', 'r')
            class_indict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)

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
            true_id = get_key(class_indict, truth)
            print(true_id)
            if true_id == -1:
                continue
            trueList.append(int(true_id))
            predictList.append(int(predict_cla))

            top5_id_numpy = top5_id
            top5_real_id_numpy = []
            for i in top5_id_numpy:
                top5_real_id_numpy.append(class_indict[str(i)])
            print(top5_real_id_numpy)
            # count += 1

            # pre_message = 'pred: ' + pred
            # truth_message = 'truth: ' + truth
            # logging.info(pre_message)
            # logging.info(truth_message)
            # logging.info(filename)
            # logging.info(top5_real_id_numpy)
            if pred != truth:
                count += 1
                error_pills.add(truth)
                #     error_pills.add(truth)
                pre_message = 'pred: ' + pred
                truth_message = 'truth: ' + truth
                logging.info(pre_message)
                logging.info(truth_message)
                logging.info(filename)
                logging.info(top5_real_id_numpy)
            if truth not in top5_real_id_numpy:
                top5_count += 1

    class_names = list(class_indict.keys())
    cf_matrix = confusion_matrix(trueList, predictList)
    df_cm = pd.DataFrame(cf_matrix, class_names,
                         class_names)  # https://sofiadutta.github.io/datascience-ipynbs/pytorch/Image-Classification-using-PyTorch.html

    ndarray = np.asarray(df_cm)

    os.makedirs('../csv', exist_ok=True)
    df_cm.to_csv('../csv/confusion_pill_0511.csv')

    # logging.info('confusion matrix pair')
    # for i in range(len(ndarray[0])):
    #     if ndarray[i][i] != 30:
    #         for j in range(len(ndarray[1])):
    #             if ndarray[i][j] != 0 and i != j:
    #                 logging.info(i)
    #                 logging.info(j)
    #                 logging.info(ndarray[i][j])

    plt.figure(figsize=(9, 6))
    # sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    sns.heatmap(df_cm, annot=False, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.savefig("../confusion_matrix_pill/confusion_matrix_pill_0517.png")
    test_dir = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill2/test/'
    test_data_length = len(os.listdir(test_dir))

    print('total test data length: ', test_data_length)
    print('error predict not in top-1:', count)
    print('error predict not in top-5:', top5_count)
    accuracy = ((test_data_length - count) / test_data_length) * 100
    top5_accuracy = ((test_data_length - top5_count) / test_data_length) * 100
    print('test accuracy:' + str(accuracy) + '%')
    print('top-5 test accuracy:' + str(top5_accuracy) + '%')
    print('error pill classes: ', error_pills)
    print('error pill classes number:', len(error_pills))
    logging.info(error_pills)
    return model


predict_capsule_id()


# layer = model.features.denseblock4.denselayer16.relu2
# layer = model.features.norm5
# https://stackoverflow.com/questions/69023069/how-to-extract-the-feature-vectors-and-save-them-in-densenet121