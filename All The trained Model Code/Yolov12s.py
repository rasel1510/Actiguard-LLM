
import subprocess
subprocess.check_call(['pip', 'install', 'ultralytics'])

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF





# Feature Enhancement Module (FEM) with atrous conv + RFB style
class FEM(nn.Module):
    def __init__(self, c1, c2, dilation_rates=[1,2,3,4]):
        super(FEM, self).__init__()
        self.conv1 = Conv(c1, c2, k=1)
        self.atrous_convs = nn.ModuleList([Conv(c2, c2, k=3, p=d, d=d, act=True) for d in dilation_rates])
        self.rfb_conv = Conv(c2 * len(dilation_rates), c2, k=1, act=True)
        
    def forward(self, x):
        x = self.conv1(x)
        out = [conv(x) for conv in self.atrous_convs]
        x = torch.cat(out, dim=1)
        x = self.rfb_conv(x)
        return x

# Custom Fusion Block (C2f + Conv)
class CustomFusion(nn.Module):
    def __init__(self, c1, c2):
        super(CustomFusion, self).__init__()
        self.c2f = C2f(c1, c2, n=1, shortcut=False)
        self.conv = Conv(c2, c2, k=1, act=True)
    def forward(self, x):
        x = self.c2f(x)
        x = self.conv(x)
        return x

# Feature Fusion Module (FFM) with BiFPN
class FFM(nn.Module):
    def __init__(self, channels_list, out_c):
        super(FFM, self).__init__()
        self.crc_convs = nn.ModuleList([Conv(c, out_c, k=1, act=True) for c in channels_list])
        self.fusion_blocks = nn.ModuleList([CustomFusion(out_c*2, out_c) for _ in range(len(channels_list)-1)])
        self.weights = nn.Parameter(torch.ones(len(channels_list)))
        
    def forward(self, features):
        crc_features = [conv(f)*w for f, conv, w in zip(features, self.crc_convs, self.weights)]
        fused = crc_features[0]
        for i in range(1, len(crc_features)):
            f_i = crc_features[i]
            if fused.shape[2:] != f_i.shape[2:]:
                f_i = F.interpolate(f_i, size=fused.shape[2:], mode='nearest')
            fused = self.fusion_blocks[i-1](torch.cat([fused, f_i], dim=1))
        return fused


# Dataset YAML

DATA_YAML = "/kaggle/input/violence-data/violenceData/data.yaml"
if not os.path.exists(DATA_YAML):
    raise FileNotFoundError("Dataset YAML not found!")


print("Loading YOLOv12s model...")
model = YOLO("yolov12s.pt")

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
    name = "yolov12s_FMF_violence_detection",
    exist_ok = True,
    box = 7.5,
    cls = 0.5,
    dfl = 1.5
)


# Training
print("Starting YOLOv12s + FEM+CRC + FFM+BiFPN training...")
results = model.train(**train_config)
print("Training Completed!")


BEST_MODEL = f"{train_config['project']}/{train_config['name']}/weights/best.pt"
if os.path.exists(BEST_MODEL):
    print(f"Evaluating Best Model at {BEST_MODEL} ...")
    metrics = model.val(data=DATA_YAML, split="val", imgsz=640, batch=16)
    print(f"Precision: {metrics.box.precision:.4f}")
    print(f"Recall: {metrics.box.recall:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
else:
    print("Best model not found!")

# Save Training Summary

import json
summary = {
    "model": "YOLOv12s with FEM+CRC and FFM+BiFPN",
    "dataset": "Violence Detection",
    "classes": ["violence","non-violence"],
    "epochs": train_config['epochs'],
    "img_size": train_config['imgsz'],
    "best_model_path": BEST_MODEL
}

with open("/kaggle/working/training_summary_FMF.json","w") as f:
    json.dump(summary, f, indent=4)

print("Training Summary Saved Successfully!")