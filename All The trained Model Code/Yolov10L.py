# Install Ultralytics
import subprocess
subprocess.check_call(['pip', 'install', 'ultralytics'])

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, Bottleneck, SPPF



class FEM(nn.Module):
    def __init__(self, c1, c2, dilation_rates=[1,2,3,4]):
        super(FEM, self).__init__()
        self.conv1 = Conv(c1, c2, k=1)
        self.atrous_convs = nn.ModuleList([Conv(c2, c2, k=3, p=r, d=r, act=True) for r in dilation_rates])
        self.rfb_conv = Conv(c2 * len(dilation_rates), c2, k=1, act=True)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([conv(x) for conv in self.atrous_convs], dim=1)
        x = self.rfb_conv(x)
        return x

class CustomFusion(nn.Module):
    def __init__(self, c1, c2):
        super(CustomFusion, self).__init__()
        self.csp = C2f(c1, c2, n=1, shortcut=False)
        self.cbs = Conv(c2, c2, k=1, act=True)
    def forward(self, x):
        return self.cbs(self.csp(x))

class FFM(nn.Module):
    def __init__(self, channels_list, c2):
        super(FFM, self).__init__()
        self.crc_conv = nn.ModuleList([Conv(c1, c2, k=1, act=True) for c1 in channels_list])
        self.fusion_blocks = nn.ModuleList([CustomFusion(c2*2, c2) for _ in range(len(channels_list)-1)])
        self.weights = nn.Parameter(torch.ones(len(channels_list)))
    def forward(self, features):
        crc_features = [conv(f)*self.weights[i] for i, (conv,f) in enumerate(zip(self.crc_conv,features))]
        fused = crc_features[0]
        for i in range(1,len(crc_features)):
            f = crc_features[i]
            if fused.shape[2:] != f.shape[2:]:
                f = F.interpolate(f, size=fused.shape[2:], mode='nearest')
            fused = self.fusion_blocks[i-1](torch.cat([fused,f],dim=1))
        return fused

# Dataset YAML

DATA_YAML = "/kaggle/input/violence-data/violenceData/data.yaml"
if not os.path.exists(DATA_YAML):
    raise Exception("Dataset YAML not found!")


model = YOLO("yolov10l.pt")  


c1 = model.model.backbone[-1].c1 if hasattr(model.model.backbone[-1],'c1') else 1024
c2 = 1024
fem_module = FEM(c1, c2)
model.model.backbone.append(fem_module)

channels_list = [256, 512, 1024]
ffm_module = FFM(channels_list, c2=512)
model.model.head.append(ffm_module)


# Training Configuration

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
    name = "yolov10l_fem_ffm",
    exist_ok = True,
    box = 7.5,
    cls = 0.5,
    dfl = 1.5
)

# Start Training

results = model.train(**train_config)

# Best Model Path

BEST_MODEL = "/kaggle/working/training_results/yolov10l_fem_ffm/weights/best.pt"

if os.path.exists(BEST_MODEL):
    print("Best model saved at:", BEST_MODEL)
else:
    print("Best model not found!")


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

# =========================================
# Save Summary
# =========================================
import json
summary = {
    "model": "YOLOv10L with FEM+FFM",
    "dataset": "Violence Detection",
    "classes": ["violence","non-violence"],
    "epochs": 60,
    "img_size": 640,
    "best_model_path": BEST_MODEL
}

with open("/kaggle/working/training_summary.json","w") as f:
    json.dump(summary,f,indent=4)

print("Training Summary Saved. Pipeline Completed!")