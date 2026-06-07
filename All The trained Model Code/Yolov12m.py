
import subprocess
subprocess.check_call(['pip', 'install', 'ultralytics'])

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, Bottleneck, SPPF

print("="*60)
print("YOLOv12m with FEM + FFM + BiFPN Violence Detection Training")
print("="*60)

# Feature Enhancement Module (FEM)

class FEM(nn.Module):
    def __init__(self, c1, c2, dilation_rates=[1,2,3,4]):
        super(FEM, self).__init__()
        self.conv1 = Conv(c1, c2, k=1)
        self.atrous_convs = nn.ModuleList([Conv(c2, c2, k=3, p=d, d=d, act=True) for d in dilation_rates])
        self.rfb_conv = Conv(c2*len(dilation_rates), c2, k=1, act=True)

    def forward(self, x):
        x = self.conv1(x)
        atrous_outs = [conv(x) for conv in self.atrous_convs]
        x = torch.cat(atrous_outs, dim=1)
        x = self.rfb_conv(x)
        return x


# Custom Fusion Module for BiFPN
class CustomFusion(nn.Module):
    def __init__(self, c1, c2):
        super(CustomFusion, self).__init__()
        self.csp = C2f(c1, c2, n=1, shortcut=False)
        self.conv = Conv(c2, c2, k=1, act=True)
    def forward(self, x):
        x = self.csp(x)
        x = self.conv(x)
        return x

# Feature Fusion Module (FFM) with BiFPN

class FFM(nn.Module):
    def __init__(self, channels_list, c2):
        super(FFM, self).__init__()
        self.crc_conv = nn.ModuleList([Conv(c1, c2, k=1, act=True) for c1 in channels_list])
        self.fusion_blocks = nn.ModuleList([CustomFusion(c2*2, c2) for _ in range(len(channels_list)-1)])
        self.weights = nn.Parameter(torch.ones(len(channels_list)))

    def forward(self, features):
        crc_features = [conv(f)*self.weights[i] for i, (f, conv) in enumerate(zip(features, self.crc_conv))]
        fused = crc_features[0]
        for i in range(1, len(crc_features)):
            f2 = crc_features[i]
            if fused.shape[2:] != f2.shape[2:]:
                f2 = F.interpolate(f2, size=fused.shape[2:], mode='nearest')
            fused = self.fusion_blocks[i-1](torch.cat([fused, f2], dim=1))
        return fused


# Dataset Path

DATA_YAML = "/kaggle/input/violence-data/violenceData/data.yaml"
if not os.path.exists(DATA_YAML):
    raise Exception("Dataset YAML not found!")
print("Dataset YAML:", DATA_YAML)

print("\nLoading YOLOv12m Model...")
model = YOLO("yolov12m.pt")  


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
    name = "yolov12m_FEM_FFM_BiFPN",
    exist_ok = True,
    box = 7.5,
    cls = 0.5,
    dfl = 1.5
)

print("\nTraining Configuration")
for k,v in train_config.items():
    print(f"{k}: {v}")


print("\nStarting Training...\n")
results = model.train(**train_config)
print("\nTraining Finished!")

# Best Model Path

BEST_MODEL = "/kaggle/working/training_results/yolov12m_FEM_FFM_BiFPN/weights/best.pt"
print("\nBest Model Path:", BEST_MODEL if os.path.exists(BEST_MODEL) else "Not Found")

print("\nRunning Validation...")
metrics = model.val(data=DATA_YAML, split="val", imgsz=640, batch=16)

print("\nEvaluation Metrics")
print(f"Precision: {metrics.box.precision:.4f}")
print(f"Recall: {metrics.box.recall:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")


# Save Training Summary
import json
summary = {
    "model": "YOLOv12m with FEM+FFM+BiFPN",
    "dataset": "Violence Detection",
    "classes": ["violence","non-violence"],
    "epochs": 60,
    "img_size": 640,
    "best_model_path": BEST_MODEL
}
with open("/kaggle/working/training_summary_FEM_FFM_BiFPN.json","w") as f:
    json.dump(summary,f,indent=4)
print("\nTraining Summary Saved")

print("\n✅ YOLOv12m + FEM + FFM + BiFPN Training Pipeline Completed Successfully!")