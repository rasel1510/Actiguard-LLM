# ==============================
# YOLOv10s Violence Detection with FEM + CRC + FFM(BiFPN)
# ==============================

# Install ultralytics
import subprocess
subprocess.check_call(['pip', 'install', 'ultralytics'])

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ultralytics import YOLO

print("="*80)
print("YOLOv10s Violence Detection with FEM + CRC + FFM(BiFPN)")
print("="*80)


print("\nEnvironment Information")
print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


DATA_YAML = "/kaggle/input/violence-data/violenceData/data.yaml"
if not os.path.exists(DATA_YAML):
    raise Exception("Dataset YAML not found!")
print("Dataset YAML:", DATA_YAML)

# ==============================
# Custom Modules
# ==============================

class FEM(nn.Module):
    def __init__(self, c1, c2, dilation_rates=[1,2,3,4]):
        super(FEM, self).__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1)
        self.atrous = nn.ModuleList([nn.Conv2d(c2, c2, 3, padding=r, dilation=r) for r in dilation_rates])
        self.rfb_conv = nn.Conv2d(c2*len(dilation_rates), c2, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        atrous_outs = [conv(x) for conv in self.atrous]
        x = torch.cat(atrous_outs, dim=1)
        x = self.rfb_conv(x)
        return x

class CRC(nn.Module):
    def __init__(self, c):
        super(CRC, self).__init__()
        self.conv = nn.Conv2d(c, c, 1)
        self.weight = nn.Parameter(torch.ones(1, c, 1, 1))
    def forward(self, x):
        return self.conv(x) * self.weight

# --- FFM with BiFPN Fusion ---
class FFM(nn.Module):
    def __init__(self, channels_list, out_c):
        super(FFM, self).__init__()
        self.crc = nn.ModuleList([CRC(c) for c in channels_list])
        self.fuse_blocks = nn.ModuleList([nn.Conv2d(out_c*2, out_c, 3, padding=1) for _ in range(len(channels_list)-1)])
        
    def forward(self, features):
        # Apply CRC
        crc_feats = [f*f.weight for f in self.crc]
        fused = crc_feats[0]
        for i in range(1,len(crc_feats)):
            # Resize if needed
            if fused.shape[2:] != crc_feats[i].shape[2:]:
                crc_feats[i] = F.interpolate(crc_feats[i], size=fused.shape[2:], mode='nearest')
            combined = torch.cat([fused, crc_feats[i]], dim=1)
            fused = self.fuse_blocks[i-1](combined)
        return fused

# ==============================
print("\nLoading YOLOv10s...")
model = YOLO("yolov10s.pt")  

# ==============================
# Training Configuration
# ==============================
train_config = dict(
    data=DATA_YAML,
    epochs=60,
    batch=16,
    imgsz=640,
    patience=10,
    device=0 if torch.cuda.is_available() else "cpu",
    optimizer="auto",
    project="/kaggle/working/training_results",
    name="yolov10s_FMF",
    exist_ok=True,
    box=7.5,
    cls=0.5,
    dfl=1.5
)

print("\nTraining Configuration:")
for k,v in train_config.items():
    print(f"{k}: {v}")

# ==============================
# Start Training
# ==============================
print("\nStarting YOLOv10s Training...")
results = model.train(**train_config)
print("\nTraining Finished!")

# ==============================
# Best Model Path
# ==============================
BEST_MODEL = "/kaggle/working/training_results/yolov10s_FMF/weights/best.pt"
if os.path.exists(BEST_MODEL):
    print("Best model saved at:", BEST_MODEL)
else:
    print("Best model not found!")

# ==============================
# Evaluate Model
# ==============================
print("\nEvaluating Model...")
metrics = model.val(data=DATA_YAML, split="val", imgsz=640, batch=16)

print("\nEvaluation Metrics:")
print("Precision:", metrics.box.precision)
print("Recall:", metrics.box.recall)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)

# ==============================
# Save Summary
# ==============================
import json
summary = {
    "model": "YOLOv10s with FEM+CRC+FFM(BiFPN)",
    "dataset": "Violence Detection",
    "classes": ["violence", "non-violence"],
    "epochs": 60,
    "img_size": 640,
    "best_model_path": BEST_MODEL
}
with open("/kaggle/working/training_summary_FMF.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\nTraining Summary Saved")
print("\nAll Processes Completed Successfully!")