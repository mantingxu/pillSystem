from PIL import Image
from torchvision import transforms
import torch
import glob
import numpy as np
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import euclidean

# model related
path = "/home/wall/finalSystem/2-densenet-capsule/weight/capsule_accuracy_0423_append_train.pth"
model = torch.load(path)
model.eval()
img_list = []
vector_list = []


# resolve problem
# pred: 12448
# truth: 12222
# /media/wall/4TB_HDD/full_dataset/0423_dataset/test_capsule_sharpen/12222_23-0.png
# ['12448', '12222', '12083']


def get_vector(im_path):
    img = Image.open(im_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to('cuda')

    # Pass input image through feature extractor
    with torch.no_grad():
        feature_map = model.features(img_tensor)  # denseNet
        feature_map = feature_map.mean(dim=1, keepdim=True)  # 取平均值，得到單通道的特徵地圖 (1, C, H, W) =>  (1, 1, H, W)
        feature_vector = feature_map.view(1, -1)  # 將 feature_map 攤平成一個一維向量

        return feature_vector


def k_nearest_neighbors(X, query, k):
    # Calculate Euclidean distances between query and all vectors in X
    distances = [euclidean(query, x) for x in X]
    # Sort distances in ascending order and get indices of the k nearest neighbors
    indices = np.argsort(distances)[:k]

    # Return the indices of the k nearest neighbors
    return indices


def get_top3_DBL(pillID):
    # image related
    image_folder = '/home/wall/finalSystem/6-DBL/dataset/train/{pillID}/*.png'.format(pillID=pillID)
    for image_path in glob.glob(image_folder):
        feature_vector = get_vector(image_path)
        vector_list.append(feature_vector)
        img_list.append(image_path)

    # target image
    query_pic_path = '/home/wall/finalSystem/6-DBL/dataset/test/12222/12222_23-0.png'
    query_pic_vector = get_vector(query_pic_path)

    all_images_vectors = []
    for vector in vector_list:
        element = vector.cpu()
        element = element.numpy()
        all_images_vectors.append(element)

    query_image_vector = query_pic_vector.cpu().numpy()

    k = 3
    nearest_neighbors = k_nearest_neighbors(all_images_vectors, query_image_vector, k)

    # get knn avg image vector
    vectors = []
    for idx in nearest_neighbors:
        vectors.append(all_images_vectors[idx])
    avg_vector = np.mean(vectors, axis=0)

    # hadamard layer
    hadamard = np.multiply(avg_vector, query_image_vector)

    # tanh layer
    hadamard_tensor = torch.from_numpy(hadamard)
    tanh = nn.Tanh() # -1~1
    output = tanh(hadamard_tensor)
    return output


vector_12222 = get_top3_DBL(12222)
vector_12448 = get_top3_DBL(12448)
vector_12083 = get_top3_DBL(12083)

one_tensor = torch.cat((vector_12222, vector_12448), 1)
final_tensor = torch.cat((one_tensor, vector_12083), 1)
label_tensor = np.array([1, 0, 0])

# create mlp model

# create dataset for ['12448', '12222', '12083']

# 1. choose target pill image(fx)
# 2. get top3 pills id
# 3. find knn (k=3)? ==> 3 similar images with target
# 4. get 3 images avg => f_pillID
# 5. create mlp dataset
# input: [input (fx, f_pillID) ==> DBL output (vector) ==> concat top3 dbl output (1*3d)]
# output => shape(1*3) ==> [0,0,1]
