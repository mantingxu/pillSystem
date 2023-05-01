from PIL import Image
from torchvision import transforms
import torch

import numpy as np
import torch.nn as nn
path = "/home/wall/finalSystem/2-densenet-capsule/weight/capsule_accuracy_0423_append_train.pth"
model = torch.load(path)

model.eval()
image_path = '/home/wall/finalSystem/6-DBL/dataset/test/12222/12222_0-0.png'
img = Image.open(image_path)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img_tensor = transform(img).unsqueeze(0).to('cuda')

# Pass input image through feature extractor
with torch.no_grad():
    feature_map = model.features(img_tensor)
    feature_map = feature_map.mean(dim=1, keepdim=True) # 取平均值，得到單通道的特徵地圖
    print(feature_map)
    print(feature_map.shape)
    m = nn.Tanh()
    output = m(feature_map).cpu().numpy()
    np.set_printoptions(suppress=True)
    print(output)
    feature_vector = feature_map.view(1, -1)
    print(feature_vector.shape)