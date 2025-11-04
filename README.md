# Visión por computador - Práctica IV
## Autores
 - Juan Carlos Rodríguez Ramírez
 - Mohamed O. Haroun Zarkik

## Introducción

## Entorno y librerías
Para el funcionamiento de esta práctica será necesario tener mucha paciencia para instalar todas las dependencias necesarias en el o los entornos. 

## Tarea I
Este proyecto desarrolla un prototipo para el procesamiento de vídeo que permite:

- Detectar y seguir personas y vehículos presentes en el vídeo.
- Detectar las matrículas de los vehículos.
- Contar el total de instancias de cada clase.
- Generar un vídeo anotado visualmente con los resultados de detección y seguimiento.
- Crear un archivo CSV con el detalle de detección y seguimiento, con los campos:
  `fotograma, tipo_objeto, confianza, identificador_tracking, x1, y1, x2, y2, matrícula_en_su_caso, confianza_matricula, mx1, my1, mx2, my2, texto_matricula`.

## Entrenamiento de Modelos
Para este proyecto se entrenaron dos modelos YOLOv11:

- **YOLOv11 Nano**: diseñado para detecciones rápidas, es un modelo muy ligero y eficiente para dispositivos con recursos limitados.
- **YOLOv11 Small**: un modelo un poco más pesado, con una arquitectura y número de parámetros superiores que permiten mayor precisión.

Ambos modelos fueron entrenados usando el mismo [código](training/train_slp.py) base y conjuntos de hiperparámetros. Para optimizar estos últimos, se utilizó un [código](training/tunning_slp.py) con la función model.tune de YOLO, que facilita la búsqueda de los hiperparámetros óptimos según el dataset empleado. Se limitaron las iteraciones a 20 para mantener una búsqueda eficaz pero no excesivamente exhaustiva.

## Dataset
El dataset fue construido combinando imágenes propias junto con imágenes tomadas de varios datasets, entre ellos uno de [Roboflow](https://universe.roboflow.com/licenseplates-h9qfr/spanish-license-plates). Actualmente, el [dataset propio](https://www.kaggle.com/datasets/juanrodrguez215/spanish-plates) está disponible en Kaggle de manera pública. Ambos modelos fueron entrenados con el mismo dataset sin ninguna variación.

## Resultados
https://github.com/user-attachments/assets/a1adb621-ea52-451d-9fcd-e1c7b9bfc8cc

Sorprendentemente, el modelo Nano parece ser ligeramente mejor, dudando menos en etiquetar una instancia y haciendo un buen seguimiento a diferencia del modelo Small. No obstante, con las matrículas se ven las carencias del modelo Nano, detectando matrículas fantasmas. Con la tarea II se verán otras comparativas interesantes, pero con respecto a esta tarea, se puede dar al modelo Nano como ganador.

---

## Tarea II


