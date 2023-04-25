from PIL import Image
from torchvision import transforms
import torch
import logging
import json
import os


def predict_capsule_id(resize_dir, capsule_class_txt):
    exp_folder = resize_dir.split('/')[-1]
    logger_path = '../2-densenet-capsule/logger/' + exp_folder
    if not os.path.isdir(logger_path):
        os.makedirs(logger_path)

    # clear handler
    logging.getLogger('').handlers = []
    logger_file = logger_path + '/logger.log'
    logging.basicConfig(filename=logger_file, level=logging.INFO, format="%(message)s", filemode="w")

    with open(capsule_class_txt, 'r') as f:
        capsule_lines = f.readlines()
    for filename in capsule_lines:
        filename = filename.replace('\n', '')
        input_image = Image.open(filename)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # data preprocess
        input_tensor = preprocess(input_image)

        # unsqueeze (batch, channel, width, height)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        best_wight_path = '../2-densenet-capsule/weight/best_accuracy_capsule.pth'
        model = torch.load(best_wight_path)

        # predict model
        model.eval()

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        try:
            json_file = open('../2-densenet-capsule/label/capsule_class_indices.json', 'r')
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
            pre_message = 'predict: ' + pred
            logging.info(pre_message)

            top5_id_numpy = top5_id
            top5_real_id_numpy = []
            for i in top5_id_numpy:
                top5_real_id_numpy.append(class_indict[str(i)])
            # print(top5_real_id_numpy)
            # print(top5_prob)
            logging.info('top-5 real id')
            logging.info(top5_real_id_numpy)
            logging.info('top-5 probability')
            logging.info(top5_prob)
    return logger_file
