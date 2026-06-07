
import subprocess
subprocess.check_call(['pip', 'install', 'ultralytics'])

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF



# Feature Enhancement Module (FEM)
class FEM(nn.Module):
    def __init__(self, c1, c2, dilation_rates=[1,2,3,4]):
        super(FEM, self).__init__()
        self.conv1 = Conv(c1, c2, k=1)
        self.atrous_convs = nn.ModuleList([Conv(c2, c2, k=3, p=r, d=r, act=True) for r in dilation_rates])
        self.rfb_conv = Conv(c2*len(dilation_rates), c2, k=1, act=True)
    
    def forward(self, x):
        x = self.conv1(x)
        outputs = [conv(x) for conv in self.atrous_convs]
        x = torch.cat(outputs, dim=1)
        x = self.rfb_conv(x)
        return x


class CustomFusion(nn.Module):
    def __init__(self, c1, c2):
        super(CustomFusion, self).__init__()
        self.csp = C2f(c1, c2, n=1, shortcut=False)
        self.conv = Conv(c2, c2, k=1, act=True)
    def forward(self, x):
        x = self.csp(x)
        x = self.conv(x)
        return x

# Feature Fusion Module (FFM)
class FFM(nn.Module):
    def __init__(self, channels_list, out_channels):
        super(FFM, self).__init__()
        self.crc_convs = nn.ModuleList([Conv(c, out_channels, k=1, act=True) for c in channels_list])
        self.fusions = nn.ModuleList([CustomFusion(out_channels*2, out_channels) for _ in range(len(channels_list)-1)])
        self.weights = nn.Parameter(torch.ones(len(channels_list)))
    
    def forward(self, features):
        # Apply CRC weighting
        crc_features = [conv(f)*self.weights[i] for i, (conv, f) in enumerate(zip(self.crc_convs, features))]
        fused = crc_features[0]
        for i in range(1, len(crc_features)):
            f = crc_features[i]
            if fused.shape[2:] != f.shape[2:]:
                f = F.interpolate(f, size=fused.shape[2:], mode='nearest')
            fused = self.fusions[i-1](torch.cat([fused,f],dim=1))
        return fused

# Dataset Configuration
DATA_YAML = "/kaggle/input/violence-data/violenceData/data.yaml"

if not os.path.exists(DATA_YAML):
    raise Exception("Dataset YAML not found!")


# Loading  YOLOv11L Model
print("Loading YOLOv11L model...")
model = YOLO("yolov11l.pt") 

train_config = dict(
    data = DATA_YAML,
    epochs = 60,
    imgsz = 640,
    batch = 8, 
    patience = 10,
    device = 0 if torch.cuda.is_available() else "cpu",
    workers = 2,
    optimizer = "auto",
    project = "/kaggle/working/training_results",
    name = "yolov11l_fmf_violence_detection",
    exist_ok = True,
    box = 7.5,
    cls = 0.5,
    dfl = 1.5
)

print("\nTraining Configuration:")
for k,v in train_config.items():
    print(f"{k}: {v}")

# # Train Model
print("\nStarting Training...\n")
results = model.train(**train_config)
print("\nTraining Completed!")

# Save Best Model

BEST_MODEL = "/kaggle/working/training_results/yolov11l_fmf_violence_detection/weights/best.pt"
if os.path.exists(BEST_MODEL):
    print("\nBest Model Found:", BEST_MODEL)
else:
    print("Best model not found!")


print("\nRunning Validation...")
metrics = model.val(
    data = DATA_YAML,
    split = "val",
    imgsz = 640,
    batch = 8
)

print("\nEvaluation Metrics:")
print("Precision:", metrics.box.precision)
print("Recall:", metrics.box.recall)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)

# Save Training Summary

import json
summary = {
    "model": "YOLOv11L with FEM + FFM",
    "dataset": "Violence Detection",
    "classes": ["violence","non-violence"],
    "epochs": 60,
    "img_size": 640,
    "best_model_path": BEST_MODEL
}

with open("/kaggle/working/training_summary.json","w") as f:
    json.dump(summary,f,indent=4)

print("\nTraining Summary Saved Successfully!")
print("\nPipeline Completed!")