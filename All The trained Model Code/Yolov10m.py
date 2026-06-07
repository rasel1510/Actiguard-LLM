
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
        super(FEM,self).__init__()
        self.conv1 = Conv(c1,c2,k=1)
        self.atrous = nn.ModuleList([Conv(c2,c2,k=3,p=d,d=d,act=True) for d in dilation_rates])
        self.rfb = Conv(c2*len(dilation_rates),c2,k=1,act=True)
        
    def forward(self,x):
        x = self.conv1(x)
        out = [conv(x) for conv in self.atrous]
        x = torch.cat(out,dim=1)
        x = self.rfb(x)
        return x


class CustomFusion(nn.Module):
    def __init__(self,c1,c2):
        super(CustomFusion,self).__init__()
        self.csp = C2f(c1,c2,n=1,shortcut=False)
        self.conv = Conv(c2,c2,k=1,act=True)
    def forward(self,x):
        return self.conv(self.csp(x))

# -----------------------------
# Feature Fusion Module (FFM) with CRC
# -----------------------------
class FFM(nn.Module):
    def __init__(self,channels_list,c2):
        super(FFM,self).__init__()
        self.crc = nn.ModuleList([Conv(c1,c2,k=1,act=True) for c1 in channels_list])
        self.fusions = nn.ModuleList([CustomFusion(c2*2,c2) for _ in range(len(channels_list)-1)])
        self.weights = nn.Parameter(torch.ones(len(channels_list)))
        
    def forward(self,features):
        crc_features = [conv(f)*self.weights[i] for i,(f,conv) in enumerate(zip(features,self.crc))]
        fused = crc_features[0]
        for i in range(1,len(crc_features)):
            if fused.shape[2:] != crc_features[i].shape[2:]:
                resized = F.interpolate(crc_features[i],size=fused.shape[2:],mode='nearest')
            else:
                resized = crc_features[i]
            fused = self.fusions[i-1](torch.cat([fused,resized],dim=1))
        return fused

# -----------------------------
# Custom YAML for YOLOv10m (FEM + FFM + CRC)
# -----------------------------
def create_custom_yaml():
    yaml_content = """
nc: 2
names: ['violence','non-violence']

# Backbone with FEM
backbone:
  #- [-1,1,Conv,[64,3,2]]
  #- [-1,1,FEM,[128]]
  #- [-1,3,C2f,[128,True]]
  #- [-1,1,Conv,[256,3,2]]
  #- [-1,3,C2f,[256,True]]
  #- [-1,1,Conv,[512,3,2]]
  #- [-1,3,C2f,[512,True]]
  #- [-1,1,SPPF,[512,5]]

# Head with FFM + CRC
head:
  #- [-1,1,nn.Upsample,[None,2,'nearest']]
  #- [[-1,5],1,Concat,[1]]
  #- [-1,3,C2f,[256]]
  #- [[0,3,6],1,FFM,[256]]   # Multi-scale fusion with CRC
  #- [[15],1,Detect,[nc]]
"""
    path = '/kaggle/working/yolov10m_fmf.yaml'
    with open(path,'w') as f:
        f.write(yaml_content)
    return path

yaml_path = create_custom_yaml()

# -----------------------------
# Dataset Path
# -----------------------------
DATA_YAML = "/kaggle/input/violence-data/violenceData/data.yaml"
if not os.path.exists(DATA_YAML):
    raise Exception("Dataset YAML not found!")

# -----------------------------
# Load YOLOv10m
# -----------------------------
model = YOLO('yolov10m.pt')

# -----------------------------
# Training Config
# -----------------------------
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
    name = "yolov10m_fmf",
    exist_ok = True,
    box = 7.5,
    cls = 0.5,
    dfl = 1.5
)

# -----------------------------
# Train
# -----------------------------
results = model.train(**train_config)

# -----------------------------
# Save Best Model
# -----------------------------
best_model = "/kaggle/working/training_results/yolov10m_fmf/weights/best.pt"
if os.path.exists(best_model):
    print("Best model saved at:",best_model)
else:
    print("Best model not found!")

# -----------------------------
# Evaluate
# -----------------------------
metrics = model.val(
    data=DATA_YAML,
    split='val',
    imgsz=640,
    batch=16
)
print("Precision:", metrics.box.precision)
print("Recall:", metrics.box.recall)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)

# -----------------------------
# Save Summary
# -----------------------------
import json
summary = {
    "model":"YOLOv10m + FEM + FFM + CRC",
    "dataset":"Violence Detection",
    "classes":["violence","non-violence"],
    "epochs":60,
    "img_size":640,
    "best_model_path":best_model
}

with open("/kaggle/working/training_summary.json","w") as f:
    json.dump(summary,f,indent=4)

print("Training Pipeline Completed Successfully!")