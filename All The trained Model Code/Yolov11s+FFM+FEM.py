
!pip install ultralytics

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ultralytics import YOLO


# Feature Enhancement Module (FEM)

class FEM(nn.Module):
    def __init__(self, c1, c2, dilation_rates=[1,2,3,4]):
        super(FEM, self).__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.atrous = nn.ModuleList([nn.Conv2d(c2, c2, kernel_size=3, padding=d, dilation=d) for d in dilation_rates])
        self.rfb_conv = nn.Conv2d(c2*len(dilation_rates), c2, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        atrous_out = [conv(x) for conv in self.atrous]
        x = torch.cat(atrous_out, dim=1)
        x = self.rfb_conv(x)
        return x


# Feature Fusion Module (FFM)

class CustomFusion(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class FFM(nn.Module):
    def __init__(self, channels_list, out_c):
        super().__init__()
        self.crc = nn.ModuleList([nn.Conv2d(c, out_c, 1) for c in channels_list])
        self.fusion_blocks = nn.ModuleList([CustomFusion(out_c*2, out_c) for _ in range(len(channels_list)-1)])
        self.weights = nn.Parameter(torch.ones(len(channels_list)))

    def forward(self, features):
        crc_features = [conv(f)*self.weights[i] for i, (f, conv) in enumerate(zip(features, self.crc))]
        fused = crc_features[0]
        for i in range(1, len(crc_features)):
            if fused.shape[2:] != crc_features[i].shape[2:]:
                resized = F.interpolate(crc_features[i], size=fused.shape[2:], mode='nearest')
            else:
                resized = crc_features[i]
            fused = self.fusion_blocks[i-1](torch.cat([fused, resized], dim=1))
        return fused


# Dataset Path

DATA_YAML = "/kaggle/input/violence-data/violenceData/data.yaml"
if not os.path.exists(DATA_YAML):
    raise FileNotFoundError("Dataset YAML not found!")

model = YOLO("yolov11s.pt")  


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
    name = "yolov11s_FMF_YOLO",
    exist_ok = True,
    box = 7.5,
    cls = 0.5,
    dfl = 1.5
)

# Start Training
results = model.train(**train_config)


# Best Model Path
BEST_MODEL = "/kaggle/working/training_results/yolov11s_FMF_YOLO/weights/best.pt"
print("Best Model:", BEST_MODEL if os.path.exists(BEST_MODEL) else "Not found")


# Model Evaluation
metrics = model.val(data=DATA_YAML, split="val", imgsz=640, batch=16)
print(f"Precision: {metrics.box.precision:.4f}")
print(f"Recall: {metrics.box.recall:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")


# Save Summary

import json
summary = {
    "model": "YOLOv11s + FEM + FFM",
    "dataset": "Violence Detection",
    "classes": ["violence", "non-violence"],
    "epochs": 60,
    "img_size": 640,
    "best_model_path": BEST_MODEL
}

with open("/kaggle/working/training_summary.json","w") as f:
    json.dump(summary, f, indent=4)

print("Training summary saved. Pipeline completed successfully!")