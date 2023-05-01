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

# image related
for image_path in glob.glob('/home/wall/finalSystem/6-DBL/dataset/train/12222/*.png'):
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
        feature_map = feature_map.mean(dim=1, keepdim=True)  # 取平均值，得到單通道的特徵地圖
        print(feature_map.shape)
        feature_vector = feature_map.view(1, -1)
        print(feature_vector.shape)
        vector_list.append(feature_vector)
        img_list.append(image_path)

print(len(vector_list))
print(len(img_list))
print(vector_list[0])
print(img_list[0])


def k_nearest_neighbors(X, query, k):
    # Calculate Euclidean distances between query and all vectors in X
    distances = [euclidean(query, x) for x in X]
    print(distances)
    # Sort distances in ascending order and get indices of the k nearest neighbors
    indices = np.argsort(distances)[:k]

    # Return the indices of the k nearest neighbors
    return distances, indices


def get_vector(path):
    one_img = Image.open(path)
    one_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    one_img_tensor = one_transform(one_img).unsqueeze(0).to('cuda')

    # Pass input image through feature extractor
    with torch.no_grad():
        one_feature_map = model.features(one_img_tensor)
        one_feature_map = one_feature_map.mean(dim=1, keepdim=True)  # 取平均值，得到單通道的特徵地圖
        one_feature_vector = one_feature_map.view(1, -1)
        return one_feature_vector


query_pic_path = '/home/wall/finalSystem/6-DBL/dataset/test/12222/12222_0-0.png'
query_pic_vector = get_vector(query_pic_path)

all_images_vectors = []
for vector in vector_list:
    element = vector.cpu()
    element = element.numpy()
    all_images_vectors.append(element)

query_image_vector = query_pic_vector.cpu().numpy()

k = 3
distances, nearest_neighbors = k_nearest_neighbors(all_images_vectors, query_image_vector, k)

vectors = []
for idx in nearest_neighbors:
    vectors.append(all_images_vectors[idx])

avg_vector = np.mean(vectors, axis=0)
avg_vector = avg_vector.reshape(36, 1)
print(avg_vector)
query_image_vector = query_image_vector.reshape(36, 1)
print(query_image_vector)

hadamard = np.multiply(avg_vector,query_image_vector)


# avg_vector & query_image_vector => hadamard layer & tanh
print(hadamard)
C_tensor = torch.from_numpy(hadamard)
m = nn.Tanh()
output = m(C_tensor)
np.set_printoptions(suppress=True)
print(output)
