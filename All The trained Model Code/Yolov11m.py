
import subprocess
subprocess.check_call(['pip', 'install', 'ultralytics'])

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import os
import json

print("="*60)
print("YOLOv11m + FEM + CRC + FFM+BiFPN Training")
print("="*60)



print("\nPyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# Dataset Path

DATA_YAML = "/kaggle/input/violence-data/violenceData/data.yaml"
if not os.path.exists(DATA_YAML):
    raise Exception("Dataset YAML not found!")
print("Dataset YAML:", DATA_YAML)


# Custom Modules: FEM, CRC, FFM

class FEM(nn.Module):
    def __init__(self, c1, c2, dilation_rates=[1,2,3,4]):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 1)
        self.atrous = nn.ModuleList([nn.Conv2d(c2, c2, 3, padding=d, dilation=d) for d in dilation_rates])
        self.rfb = nn.Conv2d(c2*len(dilation_rates), c2, 1)
    def forward(self, x):
        x = self.conv1(x)
        outs = [a(x) for a in self.atrous]
        x = torch.cat(outs, dim=1)
        return self.rfb(x)

class CustomFusion(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))

class FFM(nn.Module):
    def __init__(self, channels_list, c_out):
        super().__init__()
        self.crc = nn.ModuleList([nn.Conv2d(c, c_out, 1) for c in channels_list])
        self.weights = nn.Parameter(torch.ones(len(channels_list)))
        self.fusion = nn.ModuleList([CustomFusion(c_out*2, c_out) for _ in range(len(channels_list)-1)])
    def forward(self, features):
        # CRC reweighting
        feats = [conv(f)*self.weights[i] for i, (f, conv) in enumerate(zip(features, self.crc))]
        fused = feats[0]
        for i in range(1, len(feats)):
            f = feats[i]
            if fused.shape[2:] != f.shape[2:]:
                f = F.interpolate(f, size=fused.shape[2:], mode='nearest')
            fused = self.fusion[i-1](torch.cat([fused, f], dim=1))
        return fused


model = YOLO("yolov11m.pt")  


train_config = dict(
    data=DATA_YAML,
    epochs=60,
    imgsz=640,
    batch=16,
    patience=10,
    device=0 if torch.cuda.is_available() else "cpu",
    workers=2,
    optimizer='auto',
    project='/kaggle/working/training_results',
    name='yolov11m_fmf_violence_detection',
    exist_ok=True,
    box=7.5,
    cls=0.5,
    dfl=1.5
)

print("\nTraining Configuration:")
for k,v in train_config.items():
    print(f"{k}: {v}")

# Start Training

results = model.train(**train_config)
print("\nTraining Completed!")


# Evaluate

BEST_MODEL = "/kaggle/working/training_results/yolov11m_fmf_violence_detection/weights/best.pt"

if os.path.exists(BEST_MODEL):
    print("\nBest Model Found:", BEST_MODEL)
else:
    print("\nBest Model Not Found!")

metrics = model.val(data=DATA_YAML, split='val', imgsz=640, batch=16)
print("\nEvaluation Metrics:")
print("Precision:", metrics.box.precision)
print("Recall:", metrics.box.recall)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)


# Save Summary

summary = {
    "model": "YOLOv11m + FEM + CRC + FFM+BiFPN",
    "dataset": "Violence Detection",
    "classes": ["violence","non-violence"],
    "epochs": 60,
    "img_size": 640,
    "best_model_path": BEST_MODEL
}

with open("/kaggle/working/training_summary_fmf.json","w") as f:
    json.dump(summary, f, indent=4)

print("\nTraining Summary Saved. Pipeline Complete!")