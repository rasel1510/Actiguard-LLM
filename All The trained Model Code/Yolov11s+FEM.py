
import subprocess
subprocess.check_call(['pip', 'install', 'ultralytics'])

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, Bottleneck, SPPF


# Feature Enhancement Module (FEM)

class FEM(nn.Module):
    def __init__(self, c1, c2, dilation_rates=[1,2,3,4]):
        super(FEM, self).__init__()
        self.conv1 = Conv(c1, c2, k=1)
        self.atrous_convs = nn.ModuleList([Conv(c2, c2, k=3, p=rate, d=rate, act=True) for rate in dilation_rates])
        self.rfb_conv = Conv(c2*len(dilation_rates), c2, k=1, act=True)
    
    def forward(self, x):
        x = self.conv1(x)
        atrous_outputs = [conv(x) for conv in self.atrous_convs]
        x = torch.cat(atrous_outputs, dim=1)
        x = self.rfb_conv(x)
        return x

# YOLOv11s Model Configuration

DATA_YAML = "/kaggle/input/violence-data/violenceData/data.yaml"

if not os.path.exists(DATA_YAML):
    raise Exception("Dataset YAML not found!")


model = YOLO("yolov11s.pt")
print("YOLOv11s loaded successfully.")


for i, layer in enumerate(model.model):
    if isinstance(layer, Conv) and layer.c1 == 3:  
        model.model[i] = nn.Sequential(layer, FEM(layer.c2, layer.c2))
        print(f"FEM module integrated at layer {i}.")
        break


train_config = dict(
    data = DATA_YAML,
    epochs = 60,
    imgsz = 640,
    batch = 16,
    patience = 10,
    device = 0 if torch.cuda.is_available() else "cpu",
    workers = 2,
    optimizer = "auto",
    project = "/kaggle/working/training_results",
    name = "yolov11s_fem_violence",
    exist_ok = True,
    box = 7.5,
    cls = 0.5,
    dfl = 1.5
)

print("Starting YOLOv11s+FEM training...")
results = model.train(**train_config)
print("Training Finished!")

# Best Model Path

BEST_MODEL = "/kaggle/working/training_results/yolov11s_fem_violence/weights/best.pt"
if os.path.exists(BEST_MODEL):
    print("Best model saved at:", BEST_MODEL)
else:
    print("Best model not found!")

# Validation

metrics = model.val(
    data = DATA_YAML,
    split = "val",
    imgsz = 640,
    batch = 16
)

print("\nEvaluation Metrics for YOLOv11s+FEM")
print("Precision:", metrics.box.precision)
print("Recall:", metrics.box.recall)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)

# Save Summary

import json
summary = {
    "model": "YOLOv11s + FEM",
    "dataset": "Violence Detection",
    "classes": ["violence","non-violence"],
    "epochs": 60,
    "img_size": 640,
    "best_model_path": BEST_MODEL
}

with open("/kaggle/working/training_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\nTraining summary saved.")
print("\nYOLOv11s+FEM pipeline completed successfully!")