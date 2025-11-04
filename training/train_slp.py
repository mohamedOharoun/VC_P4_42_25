import torch
from ultralytics import YOLO
import csv
from datetime import datetime

# Lista para almacenar métricas por época
epoch_log = []

class EpochLogger:
    def __init__(self):
        self.best_map50 = 0.
        self.patience_counter = 0

    def __call__(self, results=None, epoch=None, *args, **kwargs):
        # 'results' contiene métricas del epoch actual
        if not results or epoch is None:
            return

        # Extraer map50 (adaptar si Key es distinta según versión)
        map50 = getattr(results, 'map50', None)
        if map50 is None:
            map50 = getattr(results, 'metrics', {}).get('map50', 0)

        # Actualizar mejor valor y paciencia
        if map50 > self.best_map50:
            self.best_map50 = map50
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Guardar log (epoch empieza en 0)
        epoch_log.append({
            'epoch': epoch + 1,
            'map50': map50,
            'best_map50': self.best_map50,
            'patience_counter': self.patience_counter
        })

model = YOLO('yolo11s.pt') # también con yolo11n.pt

logger = EpochLogger()
model.add_callback('on_epoch_end', logger)

model.train(
    data="/app/SLP-personalizado/data.yaml",
    epochs=120,
    batch=-1,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    imgsz=640,
    workers=4,
    patience=10,
    # ====================================================
    # Hiperparámetros obtenidos tras el tunning del modelo
    lr0=0.00755,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.00054,
    warmup_epochs=3.96534,
    warmup_momentum=0.95,
    # ====================================================
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"training_log_{timestamp}.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['epoch', 'map50', 'best_map50', 'patience_counter']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(epoch_log)

print(f"\n✓ Registro guardado en: {csv_filename}")
print(f"Total de épocas entrenadas: {len(epoch_log)}")
