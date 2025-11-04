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

Se entrenaron dos modelos YOLOv11 para este proyecto:  
- **YOLOv11 Nano**  
- **YOLOv11 Small**  

Ambos entrenamientos incluyeron una búsqueda optimizada de hiperparámetros utilizando la función `model.tune`.

Los códigos de entrenamiento de ambos modelos están disponibles y pueden consultarse en los siguientes enlaces:  
- [Código entrenamiento YOLOv11 Nano](Entrenamiento_yolo11n.ipynb)  
- [Código entrenamiento YOLOv11 Small](Entrenamiento_yolo11s.ipynb)  

## Dataset

El dataset fue construido combinando imágenes propias junto con imágenes tomadas de un dataset disponible públicamente en Kaggle.  
Acceso al dataset utilizado: [Enlace al dataset de Kaggle](https://www.kaggle.com/) *(sustituir por URL específica del dataset)*

## Uso del Prototipo

1. Preparar el entorno con las dependencias (ver notebooks para versiones y librerías utilizadas).  
2. Descargar o entrenar modelos usando los notebooks proporcionados.  
3. Ejecutar el prototipo principal sobre el vídeo(s) de interés.  
4. Revisar la salida generada:  
   - Vídeo de resultados anotado ("resultado.mp4").  
   - Archivo CSV detallado con detecciones y métricas ("detecciones.csv").

## Resultados

El sistema puede procesar vídeos propios o el vídeo ejemplo proporcionado en la práctica, mostrando seguimiento continuo y robusto de objetos a través del tiempo, junto con detección fiable de matrículas.

---

## Tarea II


https://github.com/user-attachments/assets/a1adb621-ea52-451d-9fcd-e1c7b9bfc8cc
