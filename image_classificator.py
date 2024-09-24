import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

#Carga de imagenes 
transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),
                                                     (0.5,0.5,0.5))])
dataset = ImageFolder('./Dataset', transform = transform)
classes = dataset.classes
data_loader = DataLoader(dataset, batch_size=40, shuffle=True)

def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))

data_iter = iter(data_loader)
images, labels = next(data_iter)

fig = plt.figure(figsize=(20,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])

#Imprimir labels de imagenes del dataset
labels = {}
for label in classes:
    labels[label] = 0
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

for data in data_loader:
    img, label = data
    labels[classes[label.item()]] += 1

print(labels)

train_set, valid_set = random_split(dataset, (int(len(dataset) * 0.8) + 1, int(len(dataset) * 0.2)))

#Modelo de CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(46656, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        

