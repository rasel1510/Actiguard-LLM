# Install Ultralytics
import subprocess
subprocess.check_call(['pip', 'install', 'ultralytics'])

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ultralytics import YOLO

print("="*60)
print("YOLOv11s + FFM Violence Detection Training")
print("="*60)


# Feature Fusion Module (FFM)

class CustomFusion(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.csp = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
    def forward(self, x):
        return self.csp(x)

class FFM(nn.Module):
    def __init__(self, channels_list, out_channels):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(len(channels_list)))
        self.convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in channels_list])
        self.fusion_blocks = nn.ModuleList([CustomFusion(out_channels*2, out_channels) for _ in range(len(channels_list)-1)])

    def forward(self, features):
        # Apply weighted convs
        processed = [conv(f)*self.weights[i] for i, (f, conv) in enumerate(zip(features, self.convs))]
        fused = processed[0]
        for i in range(1, len(processed)):
            feat = processed[i]
            if fused.shape[2:] != feat.shape[2:]:
                feat = F.interpolate(feat, size=fused.shape[2:], mode='nearest')
            fused = self.fusion_blocks[i-1](torch.cat([fused, feat], dim=1))
        return fused


# Dataset Path

DATA_YAML = "/kaggle/input/violence-data/violenceData/data.yaml"
if not os.path.exists(DATA_YAML):
    raise Exception("Dataset YAML not found!")
print("Dataset YAML:", DATA_YAML)

# Load YOLOv11s

print("\nLoading YOLOv11s Model...")
model = YOLO("yolov11s.pt") 

channels_list = [128, 256, 512]  
ffm_out_channels = 256
ffm = FFM(channels_list, ffm_out_channels)
model.model.head.add_module("ffm", ffm)
print("FFM integrated into YOLOv11s head.")

# Training Configuration

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
    name = "yolov11s_ffm_violence_detection",
    exist_ok = True,
    box = 7.5,
    cls = 0.5,
    dfl = 1.5
)
print("\nTraining Configuration:")
for k,v in train_config.items():
    print(f"{k}: {v}")

# Start Training
print("\nStarting Training...")
results = model.train(**train_config)
print("\nTraining Finished!")

# Best Model Path
BEST_MODEL = "/kaggle/working/training_results/yolov11s_ffm_violence_detection/weights/best.pt"
print("\nBest Model Path:", BEST_MODEL if os.path.exists(BEST_MODEL) else "Not Found")

# Evaluate Model
print("\nRunning Validation...")
metrics = model.val(
    data = DATA_YAML,
    split = "val",
    imgsz = 640,
    batch = 16
)
print("\nEvaluation Metrics:")
print("Precision:", metrics.box.precision)
print("Recall:", metrics.box.recall)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)


# Save Training Summary
import json
summary = {
    "model": "YOLOv11s + FFM",
    "dataset": "Violence Detection",
    "classes": ["violence","non-violence"],
    "epochs": 60,
    "img_size": 640,
    "best_model_path": BEST_MODEL
}
with open("/kaggle/working/training_summary.json","w") as f:
    json.dump(summary, f, indent=4)
print("\nTraining Summary Saved")
print("\nPipeline Completed Successfully!")