# Install Ultralytics
import subprocess
subprocess.check_call(['pip', 'install', 'ultralytics'])

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF

# -----------------------------
# Feature Fusion Module (FFM) with BiFPN
# -----------------------------
class CustomFusion(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.csp = C2f(c1, c2, n=1, shortcut=False)
        self.cbs = Conv(c2, c2, k=1, act=True)
    def forward(self, x):
        x = self.csp(x)
        x = self.cbs(x)
        return x

class FFM(nn.Module):
    def __init__(self, channels_list, c2=256):
        super().__init__()
        self.crc_conv = nn.ModuleList([Conv(c, c2, k=1, act=True) for c in channels_list])
        self.fusion_blocks = nn.ModuleList([CustomFusion(c2*2, c2) for _ in range(len(channels_list)-1)])
        self.weights = nn.Parameter(torch.ones(len(channels_list)))
    def forward(self, features):
        crc_features = [conv(f)*self.weights[i] for i, (f, conv) in enumerate(zip(features, self.crc_conv))]
        fused = crc_features[0]
        for i in range(1, len(crc_features)):
            target_size = fused.shape[2:]
            resized = F.interpolate(crc_features[i], size=target_size, mode='nearest') if fused.shape[2:] != crc_features[i].shape[2:] else crc_features[i]
            combined = torch.cat([fused, resized], dim=1)
            fused = self.fusion_blocks[i-1](combined)
        return fused

# -----------------------------
# Dataset YAML
# -----------------------------
DATA_YAML = "/kaggle/input/violence-data/violenceData/data.yaml"
if not os.path.exists(DATA_YAML):
    raise Exception("Dataset YAML not found!")


model = YOLO("yolov9m.pt")

ffm_module = FFM(channels_list, c2=256)

model.model[-1] = nn.Sequential(ffm_module, model.model[-1])

print("YOLOv9m with FFM + BiFPN integrated.")

# -----------------------------
# Training Configuration
# -----------------------------
train_config = dict(
    data=DATA_YAML,
    epochs=60,
    imgsz=640,
    batch=16,
    patience=10,
    device=0 if torch.cuda.is_available() else "cpu",
    workers=2,
    optimizer="auto",
    project="/kaggle/working/training_results",
    name="yolov9m_ffm_bifpn",
    exist_ok=True,
    box=7.5,
    cls=0.5,
    dfl=1.5
)

# -----------------------------
# Start Training
# -----------------------------
results = model.train(**train_config)
print("\nTraining Completed!")

# -----------------------------
# Save Best Model Path
# -----------------------------
BEST_MODEL = "/kaggle/working/training_results/yolov9m_ffm_bifpn/weights/best.pt"
if os.path.exists(BEST_MODEL):
    print("Best Model Saved:", BEST_MODEL)
else:
    print("Best model not found!")

# -----------------------------
# Evaluate Model
# -----------------------------
metrics = model.val(
    data=DATA_YAML,
    split="val",
    imgsz=640,
    batch=16
)

print("\nEvaluation Metrics:")
print("Precision:", metrics.box.precision)
print("Recall:", metrics.box.recall)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)

# -----------------------------
# Save Summary
# -----------------------------
import json
summary = {
    "model": "YOLOv9m + FFM + BiFPN",
    "dataset": "Violence Detection",
    "classes": ["violence","non-violence"],
    "epochs": 60,
    "img_size": 640,
    "best_model_path": BEST_MODEL
}

with open("/kaggle/working/training_summary.json","w") as f:
    json.dump(summary,f,indent=4)

print("\nTraining Summary Saved Successfully!")