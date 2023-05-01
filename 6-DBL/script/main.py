import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np

model = models.resnet18(pretrained=True)
print(model)
layer = model._modules.get('avgpool')
model.eval()
# add resnet layer link (https://discuss.pytorch.org/t/add-layers-on-pretrained-model/88760/3)

# Note that the performance of your pre-trained model might differ for different input sizes.
scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)

    # 4. Define a function that will copy the output of a layer
    # def copy_data(m, i, o):
    #     my_embedding.copy_(o.data)
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))

    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding


pic_one = '../dataset/test/4061/4061_20-0.png'
pic_two = '../dataset/test/12448/12448_25-0.png'

pic_one_vector = get_vector(pic_one)
pic_two_vector = get_vector(pic_two)
print(pic_one_vector.unsqueeze(0).shape)
print(pic_two_vector.unsqueeze(0).shape)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
cos_sim = cos(pic_one_vector.unsqueeze(0),
              pic_two_vector.unsqueeze(0))
print('\nCosine similarity: {0}\n'.format(cos_sim))

print(type(pic_two_vector))
res = pic_two_vector * pic_two_vector
print(res.shape)
m = nn.Tanh()
output = m(res).numpy()
np.set_printoptions(suppress=True)
print(output)

# res1 = torch.matmul(pic_one_vector, pic_two_vector)
# print(res1)
