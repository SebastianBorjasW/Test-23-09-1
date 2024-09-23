# Test-23-09-1

### Descripción:

Se le solicita desarrollar un programa en Python que clasifique imágenes en al menos tres categorías diferentes utilizando técnicas de Transfer Learning con modelos pre-entrenados de Deep Learning. El objetivo es construir un modelo que pueda reconocer y clasificar nuevas imágenes con una precisión razonable.

### Requisitos:

#### Selección del Conjunto de Datos:

Puede utilizar conjuntos de datos públicos como:
Kaggle Datasets: Animals-10 Dataset (contiene imágenes de 10 tipos de animales).
Conjunto de Datos Personalizado: Puede recopilar imágenes de tres categorías a su elección (por ejemplo, gatos, perros y pájaros).
Asegúrese de tener un conjunto equilibrado de imágenes para cada categoría.

#### Preprocesamiento de Datos:

Organizar las imágenes en directorios separados por clase para facilitar la carga y etiquetado.
Aplicar técnicas de aumento de datos (data augmentation) como rotación, volteo y escalado para aumentar la variedad del conjunto de datos.
Redimensionar las imágenes a un tamaño adecuado para el modelo pre-entrenado seleccionado (por ejemplo, 224x224 píxeles).
Implementación del Modelo:

#### Utilizar un modelo pre-entrenado disponible en Keras o PyTorch:

Congelar las capas del modelo pre-entrenado y agregar capas densas adicionales para adaptarlo a su problema de clasificación.
Compilar el modelo con una función de pérdida adecuada (por ejemplo, categorical_crossentropy) y un optimizador como Adam.
Entrenamiento y Validación:

Dividir el conjunto de datos en conjuntos de entrenamiento y validación (por ejemplo, 80% entrenamiento y 20% validación).
Entrenar el modelo y monitorear la precisión y la pérdida en ambos conjuntos.
Ajustar hiperparámetros como la tasa de aprendizaje y el número de épocas para mejorar el rendimiento.

#### Evaluación del Modelo:

Evaluar el modelo utilizando métricas como:
Precisión (Accuracy)
Generar gráficos que muestren:
La curva de precisión y pérdida durante el entrenamiento.
La matriz de confusión para visualizar el rendimiento por clase.

#### Prueba con Nuevas Imágenes:

Probar el modelo con imágenes nuevas que no formen parte del conjunto de datos utilizado.
Mostrar las predicciones y la probabilidad asociada a cada clase.

### Entregable

Grabar un video (máximo 5 minutos) mostrando:
La explicación del enfoque y las decisiones tomadas.
El código fuente y su estructura.
La ejecución del programa y demostración de su funcionamiento con diferentes imágenes.
Los resultados obtenidos y conclusiones.

### Entrega:

Código fuente bien organizado y documentado.
Archivo README con instrucciones para ejecutar el programa y explicaciones adicionales si es necesario.
El video de demostración en un formato común (por ejemplo, MP4, AVI).

Asegúrese de que el código sea reproducible; incluya instrucciones claras sobre cómo instalar dependencias y ejecutar el programa.
Si utiliza notebooks de Jupyter, proporcione el archivo .ipynb y exporte una versión en HTML o PDF.

#### Consejos:

Comience por explorar y comprender el conjunto de datos elegido.
Verifique que las imágenes estén correctamente etiquetadas y organizadas.
Realice pruebas con una pequeña parte del conjunto de datos para agilizar el proceso de desarrollo.
Documente las decisiones tomadas y los desafíos encontrados durante el desarrollo.
