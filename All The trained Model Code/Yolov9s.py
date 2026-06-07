
import subprocess, sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())


class FEM(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = Conv(c1, c2, 1)
        self.d1 = Conv(c2, c2, 3, p=1, d=1)
        self.d2 = Conv(c2, c2, 3, p=2, d=2)
        self.d3 = Conv(c2, c2, 3, p=3, d=3)
        self.fuse = Conv(c2*3, c2, 1)

    def forward(self,x):
        x = self.conv1(x)
        f1 = self.d1(x)
        f2 = self.d2(x)
        f3 = self.d3(x)
        x = torch.cat([f1,f2,f3],1)

        return self.fuse(x)


# ================================
# BiFPN Block
# ================================
class BiFPN(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.conv = Conv(c, c, 3)

    def forward(self, x1, x2):
        if x1.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='nearest')
        x = x1 + x2
        return self.conv(x)


class FFM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.reduce = Conv(c1, c2, 1)
        self.c2f = C2f(c2, c2, n=1)

    def forward(self,x):
        x = self.reduce(x)
        return self.c2f(x)


# ================================
# Register Custom Layers
# ================================
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.nn.modules import *

globals()['FEM'] = FEM
globals()['FFM'] = FFM
globals()['BiFPN'] = BiFPN


model_yaml = """

nc: 2
names: ['violence','non-violence']

backbone:
  - [-1,1,Conv,[32,3,2]]
  - [-1,1,Conv,[64,3,2]]
  - [-1,1,FEM,[128]]
  - [-1,3,C2f,[128]]
  - [-1,1,Conv,[256,3,2]]
  - [-1,3,C2f,[256]]
  - [-1,1,Conv,[512,3,2]]
  - [-1,3,C2f,[512]]
  - [-1,1,Conv,[512,3,2]]
  - [-1,3,C2f,[512]]

head:
  - [-1,1,FFM,[256]]
  - [-1,1,nn.Upsample,[None,2,'nearest']]
  - [[-1,6],1,Concat,[1]]
  - [-1,3,C2f,[256]]

  - [-1,1,nn.Upsample,[None,2,'nearest']]
  - [[-1,4],1,Concat,[1]]
  - [-1,3,C2f,[128]]

  - [[13,16],1,Detect,[nc]]

"""

yaml_path = "/kaggle/working/yolov9s_fem_ffm_bifpn.yaml"

with open(yaml_path,"w") as f:
    f.write(model_yaml)

print("Custom model YAML saved:", yaml_path)


# ================================
# Dataset Path
# ================================
DATA_YAML = "/kaggle/input/violence-data/violenceData/data.yaml"

if not os.path.exists(DATA_YAML):
    raise Exception("Dataset YAML not found!")

print("\nLoading Custom YOLOv9s FEM+FFM+BiFPN Model")

model = YOLO(yaml_path)


# ================================
# Train Model
# ================================
results = model.train(
    data = DATA_YAML,
    epochs = 60,
    imgsz = 640,
    batch = 16,
    device = 0 if torch.cuda.is_available() else "cpu",
    workers = 2,
    optimizer = "auto",
    project = "/kaggle/working/training_results",
    name = "FMF_YOLOv9s",
    exist_ok = True,
    patience = 10,
    box = 7.5,
    cls = 0.5,
    dfl = 1.5
)

print("\nTraining Completed")


metrics = model.val(data=DATA_YAML)

print("\nEvaluation Results")

print("Precision:", metrics.box.precision)
print("Recall:", metrics.box.recall)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)


# ================================
# Best Model Path
# ================================
best_model = "/kaggle/working/training_results/FMF_YOLOv9s/weights/best.pt"

print("\nBest Model Saved At:", best_model)