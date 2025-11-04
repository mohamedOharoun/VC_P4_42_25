import torch
from ultralytics import YOLO

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

model = YOLO('yolo11s.pt') # también con yolo11n.pt

results = model.tune(
    data="/app/SLP-personalizado/data.yaml",
    epochs=120,
    patience=8,
    iterations=30,                 # Número de combinaciones a probar
    device='cuda' if torch.cuda.is_available() else 'cpu',
)
