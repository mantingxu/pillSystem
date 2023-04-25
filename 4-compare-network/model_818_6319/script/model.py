import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import config
from PIL import Image
import matplotlib.pyplot as plt

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
in_features = model.fc.in_features
num_class = 2
model.fc = nn.Linear(in_features, num_class)
print(model.fc)

# load the dataset
training_dir = config.training_dir
testing_dir = config.testing_dir
training_csv = config.training_csv
testing_csv = config.testing_csv


# preprocessing and loading the dataset
class SiameseDataset:
    def __init__(self, training_csv=None, training_dir=None, transform=None):
        # used to prepare the labels and images path
        self.train_df = pd.read_csv(training_csv)
        self.train_df.columns = ["image1", "image2", "label"]
        self.train_dir = training_dir
        self.transform = transform

    def __getitem__(self, index):
        # getting the image path
        # image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 0])
        # image2_path = os.path.join(self.train_dir, self.train_df.iat[index, 1])
        image1_path = self.train_df.iat[index, 0]
        image2_path = self.train_df.iat[index, 1]

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (
            img0,
            img1,
            torch.from_numpy(
                np.array([int(self.train_df.iat[index, 2])], dtype=np.float32)
            ),
        )

    def __len__(self):
        return len(self.train_df)


# Load the the dataset from raw image folders
siamese_dataset = SiameseDataset(
    training_csv,
    training_dir,
    transform=transforms.Compose(
        [transforms.ToTensor()]
    ),
)

train_set_size = int(len(siamese_dataset) * 0.9)
valid_set_size = len(siamese_dataset) - train_set_size
print(train_set_size)
print(valid_set_size)
train_set, val_set = torch.utils.data.random_split(siamese_dataset, [train_set_size, valid_set_size])

siamese_dataset_test = SiameseDataset(
    testing_csv,
    testing_dir,
    transform=transforms.Compose(
        [transforms.ToTensor()]
    ),
)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.model = model

    def forward(self, input1, input2):
        input0 = input1 + input2
        output0 = self.model(input0)
        return output0


# Load the dataset as pytorch tensors using dataloader
train_dataloader = DataLoader(train_set,
                              shuffle=True,
                              num_workers=8,
                              batch_size=config.batch_size)

eval_dataloader = DataLoader(val_set,
                             shuffle=True,
                             num_workers=8,
                             batch_size=4)

# Declare Siamese Network
net = SiameseNetwork().cuda()
# Declare Loss Function
criterion = nn.CrossEntropyLoss()
# Declare Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.00005)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# train the model
def train(train_dataloader):
    loss = []
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

        optimizer.zero_grad()
        output = net(img0, img1)
        label = label.squeeze(1)
        label = label.to(torch.int64)
        loss_contrastive = criterion(output, label)
        loss_contrastive.backward()
        optimizer.step()
        loss.append(loss_contrastive.item())
    loss = np.array(loss)
    return loss.mean() / len(train_dataloader)


def eval(eval_dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evalLength = eval_dataloader.__len__()
    loss = []
    count = 0
    for i, data in enumerate(eval_dataloader, 0):
        img0, img1, label = data
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
        output = net(img0, img1)
        # output = net(img0, img1)
        label = label.squeeze(1)
        label = label.to(torch.int64)
        print(label)
        loss_contrastive = criterion(output, label)
        _, predictions = output.max(1)
        print(predictions)
        loss.append(loss_contrastive.item())
        # if label.cpu() == predictions.cpu():
        #     count += 1
    loss = np.array(loss)
    # print('acc:', (count / evalLength) * 100, '%')

    return loss.mean() / len(eval_dataloader)


train_loss_history = []
eval_loss_history = []
best_eval_loss_global = 9999
best_weight_path = '../weight/model_818_6319.pth'
for epoch in range(1, config.epochs):
    best_eval_loss = 9999
    train_loss = train(train_dataloader)
    train_loss_history.append(train_loss)
    eval_loss = eval(eval_dataloader)
    eval_loss_history.append(eval_loss)

    print(f"Epoch: {epoch}")
    print(f"Training loss {train_loss}")
    print("-" * 20)
    print(f"Eval loss {eval_loss}")

    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        print("-" * 20)
        print(f"Best Eval loss {best_eval_loss}")
        torch.save(net.state_dict(), best_weight_path)
    if best_eval_loss < best_eval_loss_global:
        best_eval_loss_global = best_eval_loss

print('global best eval loss', best_eval_loss_global)
torch.save(net.state_dict(), best_weight_path)
plt.plot(np.array(train_loss_history), 'r')
plt.plot(np.array(eval_loss_history), 'b')
plt.show()

# Load the test dataset
test_dataset = SiameseDataset(
    training_csv=testing_csv,
    training_dir=testing_dir,
    transform=transforms.Compose(
        [transforms.ToTensor()]
    ),
)

test_dataloader = DataLoader(test_dataset, num_workers=6, batch_size=1, shuffle=True)
testLength = test_dataset.__len__()
model = SiameseNetwork().cuda()
model.load_state_dict(torch.load(best_weight_path))
model.eval()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
count = 0
for i, data in enumerate(test_dataloader, 0):
    x0, x1, label = data
    concat = torch.cat((x0, x1), 0)
    output = model(x0.to(device), x1.to(device))
    print('first output:' + str(output))
    _, predictions = output.max(1)
    print('second output:' + str(predictions))

    if label == predictions.cpu():
        count += 1
    print("Actual label:-", label)
    # imshow(torchvision.utils.make_grid(concat))
print('acc:', (count / testLength) * 100, '%')
print('test length: ', testLength)
