# Visión por computador - Práctica IV
## Autores
 - Juan Carlos Rodríguez Ramírez
 - Mohamed O. Haroun Zarkik

## Introducción
Esta práctica trata del aprendizaje y puesta en uso de los modelos de detección en una fase (YOLO), y del aprendizaje y uso de los modelos OCR para la detección de texto.

## Entorno y librerías
Para el funcionamiento de esta práctica será necesario tener mucha paciencia para instalar todas las dependencias necesarias en el o los entornos.

```bash
conda create -n VC_P4 python=3.10.19 -y
conda activate VC_P4
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install -c conda-forge ultralytics opencv pandas easyocr pillow -y
```

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

Como se comentaba, algunas de las imágenes del dataset fueron de cosecha propia. Por ende, tendrían que anotarse, y para ello se hizo uso de la herramienta de etiquetado [CVAT](https://www.cvat.ai/). La herramienta permite exportar las anotaciones en diferentes formatos, lo cual es una gran ventaja. Para las matrículas, se hizo la exportación en formato [YOLO](https://docs.ultralytics.com/es/datasets/detect/#usage-example_1). Para las matrículas con su contenido, en formato [ICDAR Recognition](https://docs.cvat.ai/docs/manual/advanced/formats/format-icdar/).

## Análisis del código
Es necesario comentar algunos aspectos del código empleado para la detección de instancias en el vídeo:
1. Para la detección de coches y personas, es necesario usar un modelo que sea capaz de ello, y por ende se utilizaron tanto el YOLOv11 Nano como Small. Para la detección de matrículas, se usaba el entrenado propiamente.
```python
model_objects = YOLO("yolo11n.pt")       # Modelo para personas y vehículos
model_plates = YOLO("yolo11n_best.pt")   # Modelo matrícula
```

2. Existe un problema con la detección y conteo en un vídeo, y es que muy complicado no volver a contar una instancia cada vez que se detecta, a pesar de ser la misma. Cuando un coche estacionado es detectado, se cuenta. Sin embargo, si otro objeto se interpone entre el coche y la cámara, tras volver a mantener contacto visual, el coche es contado de nuevo. Es por ello que se ha intentado implementar una técnica de IoU, para evitar contear varias veces instancias inmóviles, sin resultados muy notables.
```python
def rectangles_iou(boxA, boxB):
    # Calcula el IoU entre 2 cajas: boxA y boxB = (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

def reidentify_track(new_box, new_plate_info, frame_num, max_iou=0.5, max_frame_gap=30):
    # Intentar encontrar un track_id anterior para la nueva detección usando matrícula y posición.
    for tid, (cached_plate, last_frame, cached_box) in plate_cache.items():
        if frame_num - last_frame > max_frame_gap:
            continue  # Muy viejo, descartar
        
        # Comparar matrícula si existe
        if new_plate_info and cached_plate:
            if new_plate_info['text'] == cached_plate['text']:
                # Matricula coincide: es el mismo objeto (vehículo)
                return tid
        
        # Sin matrícula o no coincide, comparar bounding boxes (IoU)
        iou = rectangles_iou(new_box, cached_box)
        if iou > max_iou:
            return tid  # Es el mismo objeto con movimiento razonable
    
    return None  # No encontrado
```

3. Para la detección de matrículas, se ha forzado al modelo a detectarlas cada 5 segundos, otorgando esa fluidez y calidad a la detección. Con 10 o más frames de espera, la detección no es tan buena.
```python
if cls in VEHICLE_CLASSES:
    if frame_idx % 5 == 0:
       plate_info = detect_plate_in_vehicle_frame(frame, box, frame_idx)
```

## Resultados
https://github.com/user-attachments/assets/a1adb621-ea52-451d-9fcd-e1c7b9bfc8cc

### Detecciones realizadas - Conteo de clases

| Clase        | YOLOv11n | YOLOv11s |
|--------------|----------|----------|
| bus          | 5        | 2        |
| car          | 180      | 182      |
| person       | 33       | 39       |
| truck        | 7        | 24       |
| motorcycle   | 5        | 4        |
| Matrículas   | 187      | 197      |

En conclusión, el modelo Nano parece ser ligeramente mejor, dudando menos en etiquetar una instancia y haciendo un buen seguimiento a diferencia del modelo Small. En ocasiones, el modelo Small parecee estar detectando matrículas fantasmas, o de alguna manera residuales de coches que ya pasaron. No obstante, se nota alguna mejoría con respecto al Nano, pues con instancias a lejanas distancias hace un mejor tracking y no duda tanto. De manera general, parece que el IoU ayuda, aunque no demasiado, a no perder la pista de los coches y redectarlos con su ID inicial, siendo este el mayor problema de la práctica. En cómputo total, parece que el Nano ha realizado un mejor trabajo, y sorprendentemente, parece ser más fiable. En la tarea II, veremos más notablemente la importancia de detectar correctamente las matrículas para su posterior lectura.

---

## Tarea II


