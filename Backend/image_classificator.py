#Modelo de Deep Learning 
from dotenv import load_dotenv
import torch
import matplotlib
import os
import time
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
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sb
from matplotlib import style
style.use('seaborn-v0_8-whitegrid')
matplotlib.use('Agg')

#Importar librerias para iniciar un servidor
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
import traceback
import sys
import io

load_dotenv()
app = Flask(__name__)
CORS(app)


#Configuracion de logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#Carga de imagenes 
transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),
                                                     (0.5,0.5,0.5))])


#Creacion del dataset, clases y data loaders
dataset = ImageFolder('./Dataset/raw-img', transform = transform)
classes = dataset.classes
data_loader = DataLoader(dataset, batch_size=20, shuffle=True)

#Mostrar imagenes del modelo
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

#Dividir entre sets de Entrenamiento, validacion y test
train_set, test_sample = random_split(dataset, (int(len(dataset) * 0.8) + 1, int(len(dataset) * 0.2)))
train_set, valid_set = random_split(train_set, (int(len(train_set) * 0.7) + 1, int(len(train_set) * 0.3)))

#Modelo de CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(46656, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
#Crear DataLoaders para los sets de entrenamiento y validación
train_loader = DataLoader(train_set, batch_size=70)
valid_loader = DataLoader(valid_set, batch_size=1)
test_loader = DataLoader(test_sample, batch_size=1)

#Entrenamiento del modelo
def train_Model(model, train_loader, valid_loader, criterion, optimizer, device):
    total_step = len(train_loader)
    num_epochs = 10
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img = img.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(img)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img.size(0)
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
          epoch, train_loss, valid_loss))
        
    return train_losses, valid_losses

#Funcion para medir precision total del modelo
def accuracy(model, test_loader):
    correct = 0
    total = 0
    model.to("cpu")
    dataiter = iter(test_loader)
    with torch.no_grad():
        for data in dataiter:
            img, label = data
            output = model(img)
            _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    print(f"Accuracy: {100*correct / total}")

#Funcion para medir precision por clase
def accuracy_per_label(model, test_loader, classes, device):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    cnn.to(device)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = cnn(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            if(c.item()):
                class_correct[labels.item()] += 1
            class_total[labels.item()] += 1
    for i in range(10):
        print(f"{classes[i]} | Correct: {class_correct[i]} | Total: {class_total[i]}" +
              f"| Accuracy: {class_correct[i] / class_total[i]}")
        


#Hacer transfer learning con un modelo pre entrenado resnet50
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        #Congelar capas del modelo
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

    def forward(self, image):
        output = self.resnet(image)
        return output
    

#Grafica de perdida y aprendizaje
def plotGraphLearning(train_losses, valid_losses):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Entrenamiento')
    plt.plot(valid_losses, label='Validación')
    plt.title('Curva de aprendizaje')
    plt.xlabel('Época')
    plt.ylabel('Perida')
    plt.legend()
  

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

#Matriz de confusion
def plot_confusion_matrix(model, test_loader, classes, device):
    model.to(device)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            images = data.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10,7))
    sb.heatmap(cm, annot=True, fmt = 'd', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta')
    

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

#Ajustar el script para usarlo en la UI
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando: {device}")

#Cargar el modelo
try:
    model = Classifier().to(device)
    model.load_state_dict(torch.load('cnnV5.pt', map_location=device))
    model.eval()
    logger.info("Modelo cargado")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

#Endpoint para realizar inferencia de la imagen cargada
@app.route('/classify', methods=['POST'])
def predict_image():
    logger.info("Solicitud de clasificacion")
    if 'file' not in request.files:
        logger.warning("No se encontro la solicitud")
        return jsonify({'error' : 'No file'}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("Sin nombre de archivo")
        return jsonify({'error' : 'No selected file'}), 400
    
    if file:
        try:
            #leer la imagen
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            logger.info(f"Imagen leida: {file.filename}")

            image_tensor = transform(image).unsqueeze(0).to(device)
            logger.info("Imagen procesada")

            start_time = time.time()
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted = torch.argmax(probabilities).item()
                max_probability = probabilities[0][predicted].item()

            inference_time = time.time() - start_time

            logger.info(f"Prediccion: Clase {predicted}, probabilidad {max_probability}")

            all_probabilities = probabilities[0].tolist()

            #Probabilidades por clase
            class_probabilities = [
                {'class': class_name, 'probability': prob}
                for class_name, prob in zip(classes, all_probabilities)
            ]

            class_probabilities.sort(key=lambda x: x['probability'], reverse=True)
            
            #Respuesta del API con datos de inferencia y probabilidades
            response = {
                'predicted_class': classes[predicted],
                'max_probability': max_probability,
                'inference_time': inference_time,
                'class_probability': class_probabilities
            }

            #Umbral de confidencia 
            confidence = 0.9
            if max_probability < confidence:
                response['Warning'] = 'Probabilidad baja para inferir correctamente'

           
            return jsonify(response)
        except Exception as e:
            logger.error(f"Error durante la clasificacion: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)




    

