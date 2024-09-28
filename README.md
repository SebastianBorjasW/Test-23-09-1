# Test-23-09-1

# Descripción 
Este proyecto es un modelo de Deep learning creado a base del modelo pre entrenado ResNet50 de la librería de PyTorch, este modelo fue creado para clasificación de imágenes de animales. El proyecto cuenta con una UI en forma de webapp donde se puede cargar imágenes de animales y que el modelo infiera dentro de los animales que conoce (Mariposas, gatos, gallinas, vacas, elefantes, perros, caballos, ovejas, arañas y ardillas). La webapp incluye las gráficas de perdida y precisión durante el entrenamiento y la matriz de confusión del modelo. 


## Desarrollo

### Pre requisitos
- [CUDA (En caso de tener un equipo con GPU CUDA)](https://developer.nvidia.com/cuda-toolkit)
- [Python](https://www.python.org/)
- [Git](https://git-scm.com/)
- [VsCode](https://code.visualstudio.com/)
- [Dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10)


### Inicio de instalación
1. Agrega Dataset
Debido al peso de las todas las imágenes del dataset requerido, se necesitan descargar de la liga https://www.kaggle.com/datasets/alessiocorrado99/animals10 y posteriomente agregar a la carpeta "Dataset", de forma de que la ruta quede de la siguiente forma
```bash
Backend/Dataset/raw-img/...
``` 

2. Instalar
Para instalar Pytorch en Windows y Linux use el siguiente comando
```bash
pip3 install torch torchvision torchaudio

#Si se tiene una GPU con CUDA cores
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
``` 

En MacOS usar el siguiente comando
```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
``` 


Instalar las siguientes librerías en Python
```bash
pip3 install flask
pip3 install matplotlib
pip3 install flask-cors
pip3 install scikit-learn
pip3 install seaborn
```

2. Clonar el repositorio
```bash
git clone https://github.com/SebastianBorjasW/Test-23-09-1.git
```

3. Cambiar al directorio del proyecto
```bash
code Test-23-09-1
```

4. Instalar las dependencias
```bash
cd ImageClassificator
npm install
```

5. Iniciar el proyecto
```bash
cd ImageClassificator
npm run dev
```

```bash
cd Backend
python image_classificator.py
```