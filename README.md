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

Mientras que la Tarea I se centró en la detección y seguimiento de objetos (vehículos, personas y matrículas) usando YOLOv11, esta tarea aborda un desafío más específico: el **Reconocimiento Óptico de Caracteres (OCR)**.

Como se observó en los resultados de la Tarea I, los modelos YOLO son excelentes para *localizar* la matrícula, pero no para *leer* el texto que contiene. La Tarea II se enfoca en implementar, entrenar y comparar modelos diseñados específicamente para leer el texto de las matrículas detectadas.

Para ello, se utilizan dos *notebooks*:
1.  **`entrenamiento-ocr.ipynb`**: Entrena un modelo OCR personalizado (una CRNN) desde cero.
2.  **`VC_P4_B.ipynb`**: Compara el modelo personalizado contra una librería popular (EasyOCR) en un vídeo de prueba.

---

### 1. Entrenamiento del Modelo OCR (`entrenamiento-ocr.ipynb`)

Este *notebook* detalla el proceso completo de creación de un modelo OCR propio, especializado en la lectura de matrículas españolas.

#### Librerías Empleadas
* **torch / torchvision**: El framework principal para construir y entrenar la red neuronal.
    * `pip install torch torchvision`
* **pandas**: Utilizado para cargar y gestionar las etiquetas (el texto de cada matrícula) desde el archivo `gt.txt`.
    * `pip install pandas`
* **opencv-python (cv2)**: Necesario para cargar las imágenes de las matrículas y aplicar pre-procesamiento (cambio de tamaño, padding).
    * `pip install opencv-python`
* **numpy**: Para operaciones numéricas y manipulación de imágenes.
    * `pip install numpy`

#### Paso 1: Definición del Modelo (CRNN)

Se implementa una arquitectura **CRNN (Convolutional Recurrent Neural Network)**, un estándar de la industria para el reconocimiento de texto.

* **Parte Convolucional (CNN)**: Una serie de capas `Conv2d` y `MaxPool2d` actúan como un extractor de características. Aprenden a identificar patrones visuales (líneas, curvas, formas) en la imagen de la matrícula.
* **Parte Recurrente (RNN)**: La salida de la CNN se "aplana" y se pasa a una `LSTM` (Long Short-Term Memory). Esta red recurrente procesa la secuencia de características de izquierda a derecha, aprendiendo el orden y la relación entre los caracteres.
* **Capa Final**: Una capa `Linear` proyecta la salida de la LSTM al número de clases (los 37 caracteres posibles: 0-9, A-Z y el carácter *'blank'*).

```python
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # ... más capas convolucionales ...
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2,1), (2,1))
        )
        
        # 🔹 Cambiamos el input_size para que coincida con la salida real del CNN
        self.rnn = nn.LSTM(1024, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.view(b, c*h, w).permute(0, 2, 1)  # (batch, width, features)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
```

#### Paso 2: Carga y Procesamiento de Datos

Se crea una clase `PlatesDataset` personalizada que se encarga de:
1.  Leer el `gt.txt` y asociar cada imagen con su texto.
2.  Cargar cada imagen en escala de grises.
3.  **Redimensionar y rellenar (padding)**: Todas las imágenes se fuerzan a un tamaño fijo (ej. 128x32 píxeles) para que puedan procesarse en lotes (batches), manteniendo la relación de aspecto.
4.  **Codificar el texto**: Convierte el texto (ej. "4517MFC") en una secuencia de índices numéricos (ej. `[14, 15, 11, 17, 22, 15, 12]`) que la red pueda entender.

```python
class PlatesDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, max_len=10):
        # ... (inicialización) ...
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # ... (manejo de imágenes corruptas) ...

        # resize manteniendo ratio y rellenando
        h, w = img.shape
        new_h = IMG_H
        new_w = int(w * (IMG_H / h))
        new_w = min(new_w, IMG_W)
        img = cv2.resize(img, (new_w, new_h))
        if new_w < IMG_W:
            pad = np.full((IMG_H, IMG_W - new_w), 255, dtype=np.uint8)
            img = np.concatenate([img, pad], axis=1)
        
        # ... (conversión a tensor y etiquetas) ...
        
        labels = self.text_to_labels(row['text'])
        return torch.tensor(img).float(), torch.tensor(labels).int(), len(labels)
```

### Paso 3: Entrenamiento con CTCLoss

El modelo se entrena usando nn.CTCLoss (Connectionist Temporal Classification). Esta función de pérdida es fundamental para el OCR: permite al modelo aprender a predecir la secuencia de caracteres correcta sin necesidad de saber la ubicación exacta de cada letra en la imagen. Simplemente se le da la imagen y el texto final, y la CTCLoss se encarga de alinear la predicción de la red con la etiqueta real.

El dataset se divide (90% entrenamiento, 10% validación) y se entrena durante 30 épocas, guardando el modelo con la menor pérdida de validación.

2. Comparativa de Modelos OCR (VC_P4_B.ipynb)

Este notebook toma el modelo entrenado (ocr_v3.pt) y lo compara en un escenario real contra la popular librería EasyOCR.

  Librerías Empleadas

    - ultralytics: Para cargar el modelo YOLOv11 (de la Tarea I) y detectar las matrículas en el vídeo.

      - pip install ultralytics

    - easyocr: La librería de OCR pre-entrenada que usaremos como baseline para la comparación.

      - pip install easyocr

    - torch: Para cargar y ejecutar nuestro modelo CRNN personalizado.

      - pip install torch

    - pandas: Para almacenar los resultados de la comparación en un archivo CSV.

      - pip install pandas

    - opencv-python (cv2): Para leer el vídeo de entrada (plates_test.mp4) fotograma a fotograma.

      - pip install opencv-python

#### Proceso de Comparación

1. **Carga de Modelos**: Se cargan tres componentes:

      - El detector YOLO (yolo11n_best.pt).

      - El lector de EasyOCR (easyocr.Reader(['es', 'en'])).

      - Nuestro CRNN personalizado (ocr_v3.pt), junto con su definición de clase y transformaciones de imagen (escala de grises, redimensionado a 32x128).

2. **Procesamiento del Vídeo**: El script itera sobre cada fotograma del vídeo plates_test.mp4.

      - **Detección (YOLO)**: Primero, YOLO detecta la posición (x1, y1, x2, y2) de cualquier matrícula en el fotograma.

      - **Recorte (Crop)**: La región de la matrícula se recorta de la imagen original.

      - **Inferencia (OCR)**: Esta imagen recortada se envía a ambos modelos de OCR:

          - EasyOCR procesa la imagen directamente.

          - La imagen se pre-procesa (transforma) y se envía al modelo CRNN.

      - **Almacenamiento**: El texto predicho por ambos modelos se guarda en una lista.

3.  Generación de Resultados: Al finalizar el vídeo, todos los resultados se vuelcan a un archivo CSV (comparacion_ocr_v3_yolo11n.csv). Este archivo permite un análisis detallado, fotograma a fotograma, de qué modelo fue más preciso, cuántos fallos tuvo cada uno y en qué fotogramas específicos se produjeron los errores.

A continuación, se comentan las partes clave del script de comparación (`VC_P4_B.ipynb`).

#### Bloque 1: Carga de Detectores y Lectores

Antes de procesar el vídeo, se inicializan los modelos principales.

En primera instancia, se realizan las siguientes operaciones:
- Se define el device (CPU o CUDA) para la ejecución.

- Se carga el detector YOLOv11 entrenado para localizar las matrículas.

- Se inicializa el lector de EasyOCR, que será nuestro modelo base de comparación.
```python
device = "cuda" if torch.cuda.is_available() else "cpu"

# Detector YOLO (de la Tarea I)
detector = YOLO("models/yolo11n_best.pt")

# EasyOCR (idiomas español e inglés)
reader_easy = easyocr.Reader(['es', 'en'])
```

Aquí tienes el desglose final en formato Markdown:
Markdown

---

### 3. Desglose del Código de Comparación (`VC_P4_B.ipynb`)

A continuación, se comentan las partes clave del script de comparación (`VC_P4_B.ipynb`).

#### Bloque 1: Carga de Detectores y Lectores

Antes de procesar el vídeo, se inicializan los modelos principales.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

# Detector YOLO (de la Tarea I)
detector = YOLO("models/yolo11n_best.pt")

# EasyOCR (idiomas español e inglés)
reader_easy = easyocr.Reader(['es', 'en'])

    Se define el device (CPU o CUDA) para la ejecución.

    Se carga el detector YOLOv11 entrenado para localizar las matrículas.

    Se inicializa el lector de EasyOCR, que será nuestro modelo base de comparación.
```

#### Bloque 2: Carga del Modelo CRNN Personalizado

Para cargar un modelo PyTorch (.pt), es necesario tener definida su arquitectura (la clase CRNN) en el script. Para lo cual:
- Se define la clase CRNN exactamente igual que en el notebook de entrenamiento.

- Se carga el archivo .pt con los pesos entrenados y se envía al device.

- model_crnn.eval() es crucial para desactivar capas como BatchNorm o Dropout, asegurando que la inferencia sea consistente. 

Brevemente, se realizan las siguientes tareas en el siguiente fragmento de código:

- transform: Define la canalización de pre-procesamiento: escala de grises, redimensión a 32x128 píxeles y conversión a Tensor.

- CHARS: El mapa de caracteres que el modelo puede predecir.

- decode_ctc: Función clave que toma la salida de la red (una matriz de probabilidades) y la colapsa en texto legible, eliminando duplicados y caracteres "blank" (vacíos).

```python
# Transformaciones para CRNN
transform = T.Compose([
    T.Grayscale(),
    T.Resize((32, 128)), # Tamaño fijo (H, W) con el que se entrenó
    T.ToTensor(),
])

# Diccionario de caracteres (debe ser idéntico al de entrenamiento)
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
idx_to_char = {i: c for i, c in enumerate(CHARS)}

def decode_ctc(output):
    """Decodifica la salida CTC en texto."""
    pred = output.softmax(2).argmax(2).squeeze(0).cpu().numpy()
    text = ""
    prev_char = -1
    for c in pred:
        if c != prev_char and c < len(CHARS):
            text += idx_to_char.get(c, "")
        prev_char = c
    return text
```

#### Bloque 3: Transformaciones y Decodificador CTC

El modelo CRNN no acepta una imagen en crudo. Requiere transformaciones específicas y una función para decodificar su salida.

#### Bloque 4: Bucle Principal de Procesamiento de Vídeo

Esta es la sección central que itera sobre el vídeo, detecta y compara los OCR. Se puede dividir en las siguientes partes:

---

##### 4.1: Inicialización y Lectura del Vídeo

Primero, abrimos el archivo de vídeo y preparamos las variables para el bucle.

```python
VIDEO = "plates_test.mp4"
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
data_rows = []

print("Procesando vídeo...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    timestamp = datetime.fromtimestamp((frame_count / fps)).strftime("%H:%M:%S.%f")[:-3]
```

- cv2.VideoCapture(VIDEO): Abre el archivo de vídeo (plates_test.mp4).

- cap.get(cv2.CAP_PROP_FPS): Obtiene la tasa de fotogramas por segundo (FPS) del vídeo. Esto es vital para calcular el timestamp (marca de tiempo).

- data_rows = []: Inicializa la lista que almacenará todos nuestros resultados antes de guardarlos en un CSV.

- while cap.isOpened(): Inicia el bucle que se ejecutará mientras el vídeo esté abierto.

- ret, frame = cap.read(): Lee un único fotograma. ret es un booleano que indica si la lectura fue exitosa, y frame es la imagen en sí (como un array de NumPy).

- if not ret: break: Si ret es False, significa que el vídeo ha terminado, por lo que salimos del bucle.

- timestamp = ...: Calcula la marca de tiempo exacta del fotograma actual dividiendo el número de fotograma (frame_count) por los FPS.

##### 4.2: Detección y Recorte de la Matrícula

Dentro del bucle, por cada fotograma, primero usamos YOLO (de la Tarea I) para encontrar la matrícula y luego la recortamos.

```python
    # 1. Detección con YOLO
    results = detector(frame, verbose=False)

    if results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            placa = frame[y1:y2, x1:x2] # 2. Recorte de la matrícula
            if placa.size == 0:
                continue
```
- results = detector(frame, verbose=False): Pasa el fotograma completo al modelo YOLO (detector) para que encuentre objetos. verbose=False evita que imprima información de detección en la consola por cada fotograma.

- if results[0].boxes:: Comprueba si YOLO realmente detectó alguna caja (matrícula) en este fotograma.

- for box in results[0].boxes:: Itera sobre todas las matrículas encontradas (en caso de que haya más de una).

- x1, y1, x2, y2 = map(int, ...): Extrae las coordenadas de la caja detectora.

- placa = frame[y1:y2, x1:x2]: Este es el recorte. Usando slicing de NumPy, seleccionamos solo la región de interés (la matrícula) del fotograma original. Esta imagen placa es la que se usará para el OCR.

- if placa.size == 0:: Una comprobación de seguridad. Si el recorte falla y produce una imagen vacía, saltamos esta detección y continuamos con la siguiente.

##### 4.3: Inferencia con EasyOCR

Enviamos la imagen recortada (placa) al primer modelo: EasyOCR.

```python
# 3. Inferencia con EASY OCR
            try:
                text_easy = reader_easy.readtext(placa, detail=0, allowlist=CHARS)
                text_easy = max(text_easy, key=len).replace(" ", "") if text_easy else ""
            except:
                text_easy = ""
```

- try...except...: Se usa un bloque try porque el proceso de OCR puede fallar (por ejemplo, si la imagen es puro ruido). Si falla, simplemente asignamos un texto vacío "".

- reader_easy.readtext(placa, ...): Ejecuta la inferencia de EasyOCR sobre la imagen recortada.

- detail=0: Indica a EasyOCR que devuelva solo una lista de strings con el texto, en lugar de objetos con coordenadas y confianza.

- allowlist=CHARS: Una optimización clave. Restringe a EasyOCR para que solo reconozca los caracteres que le pasamos (nuestro alfabeto 0-9 y A-Z), ignorando símbolos o letras raras.

- max(text_easy, key=len)...: A veces, readtext puede devolver varios fragmentos (ej. ['4517', 'MFC']). Este código toma el fragmento más largo (o el único, si solo hay uno) y elimina los espacios.

##### 4.4: Inferencia con CRNN (Modelo Propio)

A continuación, enviamos la misma imagen recortada a nuestro modelo CRNN personalizado.

```python
# 4. Inferencia con CRNN (modelo propio)
            try:
                placa_rgb = cv2.cvtColor(placa, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(placa_rgb)
                img_t = transform(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model_crnn(img_t)
                text_crnn = decode_ctc(out)
            except:
                text_crnn = ""
```

- placa_rgb = cv2.cvtColor(placa, cv2.COLOR_BGR2RGB): OpenCV carga imágenes en formato BGR (Azul, Verde, Rojo). Las transformaciones de PyTorch (transform) esperan formato RGB. Esta línea corrige el orden de los canales de color.

- img_pil = Image.fromarray(placa_rgb): Convierte la imagen de un array de NumPy (formato OpenCV) a un objeto de imagen PIL (formato que esperan las transformaciones).

- img_t = transform(img_pil): Aplica la secuencia de transformaciones definida en el Bloque 3 (escala de grises, redimensionado a 32x128, conversión a Tensor).

- .unsqueeze(0): Nuestro modelo espera un "lote" (batch) de imágenes. Esta función añade una dimensión extra al principio, convirtiendo la forma de [Canales, Alto, Ancho] a [1, Canales, Alto, Ancho], simulando un lote de tamaño 1.

- with torch.no_grad(): Desactiva el cálculo de gradientes. Es una optimización crucial durante la inferencia, ya que reduce el uso de memoria y acelera el proceso (no estamos entrenando).

- out = model_crnn(img_t): Ejecuta la inferencia de nuestro modelo CRNN.

- text_crnn = decode_ctc(out): Usa la función auxiliar (definida en el Bloque 3) para convertir la salida cruda del modelo (probabilidades) en un string de texto limpio.

##### 4.5: Almacenamiento de Resultados

Finalmente, agrupamos los resultados de ambas inferencias y los añadimos a nuestra lista. Se crea un diccionario que contiene el número de fotograma, la marca de tiempo y el texto predicho por ambos modelos para esta detección específica. Al final del vídeo, esta lista contendrá el historial completo de todas las detecciones.

```python
# 5. Almacenamiento de resultados
            data_rows.append({
                "Frame": frame_count,
                "Tiempo": timestamp,
                "EasyOCR": text_easy,
                "CRNN_Custom": text_crnn
            })
```

#### Bloque 5: Guardado de Resultados

Al finalizar el bucle, se liberan los recursos y se guardan los datos recopilados en un archivo CSV usando pandas.

```python
cap.release()

# Guardar resultados
df = pd.DataFrame(data_rows)
df.to_csv("comparacion_ocr_v3_yolo11n.csv", index=False)
print("Comparación completada.")
```

## Análisis y comparativa de resultados
Este es un breve análisis comparativo del **rendimiento de detección** (la tasa de lecturas no nulas) de los diferentes métodos de OCR (EasyOCR, CRNN_Custom, Tesseract) basado en los tres archivos CSV proporcionados.

### Tasa de Lecturas (No Nulas vs. Placeholder '0')

La siguiente tabla resume cuántas lecturas válidas (definidas como una salida no nula o, en el caso del archivo antiguo, una salida que no sea `0`) produjo cada método.

| Archivo / Modelo | Total de Filas | Lecturas EasyOCR | Lecturas CRNN_Custom | Lecturas Tesseract |
| :--- | :---: | :---: | :---: | :---: |
| `...yolo11n.csv` (YOLOv11 Nano) | 172 | 98 (57.0%) | 172 (100.0%) | N/A |
| `...yolo11s.csv` (YOLOv11 Small) | 254 | 28 (11.0%) | 254 (100.0%) | N/A |
| `...tessaract.csv` (Antigua) | 204 | 106 (52.0%) | 0 (0.0%) | 0 (0.0%) |

---

### Conclusiones Clave

1.  **Rendimiento de `CRNN_Custom`:** Este modelo muestra dos comportamientos completamente diferentes:
    * En las pruebas con **YOLO (`...yolo11n.csv` y `...yolo11s.csv`)**, tiene una tasa de respuesta del 100%. Esto significa que *siempre* devuelve un valor.
    * En la prueba **`...tessaract.csv` (Antigua)**, el modelo usaba `0` como valor "placeholder" (marcador de posición) para indicar "no lectura", resultando en 0 lecturas válidas.

2.  **Rendimiento de `EasyOCR`:** El rendimiento de `EasyOCR` parece depender en gran medida del detector de matrículas utilizado.
    * Tuvo su peor rendimiento con el detector `YOLOv11s` (solo un 11.0% de lecturas).
    * Tuvo un rendimiento moderado con `YOLOv11n` (57.0%) y con el detector del archivo "Antiguo" (52.0%).

3.  **Rendimiento de `Tesseract`:** En el conjunto de datos "Antiguo" donde fue probado, `Tesseract` no produjo **ninguna** lectura válida (0%).

4.  **Tasa de Respuesta vs. Precisión:** Es importante notar que una "tasa de lectura" del 100% (como la de `CRNN_Custom` en los archivos YOLO) no implica un 100% de *precisión*. Simplemente significa que el modelo siempre genera una salida. Por el contrario, `EasyOCR` parece devolver un valor solo cuando detecta una matrícula con un nivel de confianza suficiente.