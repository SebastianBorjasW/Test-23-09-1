# Test-23-09-1

# Descripción 
Este proyecto es un modelo de Deep learning creado a base del modelo pre entrenado ResNet50 de la librería de PyTorch, este modelo fue creado para clasificación de imágenes de animales. El proyecto cuenta con una UI en forma de webapp donde se puede cargar imágenes de animales y que el modelo infiera dentro de los animales que conoce (Mariposas, gatos, gallinas, vacas, elefantes, perros, caballos, ovejas, arañas y ardillas). La webapp incluye las gráficas de perdida y precisión durante el entrenamiento y la matriz de confusión del modelo. 


## Desarrollo

### Pre requisitos
- [CUDA (En caso de tener un equipo con GPU CUDA)](https://developer.nvidia.com/cuda-toolkit)
- [Python](https://www.python.org/)
- [Git](https://git-scm.com/)
- [VsCode](https://code.visualstudio.com/)


### Inicio de instalación

1. Instalar
Para instalar Pytorch en Windows y Linux use el siguiente comando
```bash
pip3 install torch torchvision torchaudio

#Si se tiene una GPU con CUDA cores
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
``` 

En MacOS usar el siguiente comando
```bash
#En MacOS se usará el CPU ya que CUDA no esta disponible
pip3 install torch torchvision torchaudio
``` 


Instalar flask
```bash
pip install flask
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