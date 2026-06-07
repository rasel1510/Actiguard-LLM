!pip install -q torch torchvision \
  detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/ \
  pycocotools \
  codecarbon psutil thop scipy




import os
import time
import json
import torch
import psutil
import numpy as np
from PIL import Image
from codecarbon import EmissionsTracker
from thop import profile
from sklearn.metrics import precision_score, recall_score, f1_score

import detectron2
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader



# Registering the dataset 
import os
from pathlib import Path

def convert_yolo_to_coco(images_dir, labels_dir, output_json):
    p = Path(images_dir)
    image_files = sorted(list(p.glob("*.[jp][pn]g")))
    
    coco = {"images": [], "annotations": [], "categories": []}
    coco["categories"] = [{"id": 0, "name": "violence"}, {"id": 1, "name": "non-violence"}]

    ann_id = 0
    for img_id, img_path in enumerate(image_files):
        file_name = img_path.name
        w,h = Image.open(img_path).size

        coco["images"].append({
            "id": img_id,
            "width": w,
            "height": h,
            "file_name": file_name
        })

        label_path = Path(labels_dir)/img_path.with_suffix(".txt").name
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    cls,x_center,y_center,bw,bh = map(float,line.split())
                    x0 = (x_center - bw/2)*w
                    y0 = (y_center - bh/2)*h
                    bw_abs = bw*w
                    bh_abs = bh*h
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(cls),
                        "bbox": [x0, y0, bw_abs, bh_abs],
                        "area": bw_abs*bh_abs,
                        "iscrowd": 0
                    })
                    ann_id += 1

    with open(output_json,"w") as f:
        json.dump(coco,f)
    print("Saved COCO JSON:", output_json)


# Peparing using Json
convert_yolo_to_coco(
    "/kaggle/input/violence-data/violenceData/train/images",
    "/kaggle/input/violence-data/violenceData/train/labels",
    "/kaggle/working/train_coco.json"
)

convert_yolo_to_coco(
    "/kaggle/input/violence-data/violenceData/valid/images",
    "/kaggle/input/violence-data/violenceData/valid/labels",
    "/kaggle/working/val_coco.json"
)

convert_yolo_to_coco(
    "/kaggle/input/violence-data/violenceData/test/images",
    "/kaggle/input/violence-data/violenceData/test/labels",
    "/kaggle/working/test_coco.json"
)


# Registering the dataset to Detectron2
register_coco_instances(
    "violence_train",
    {},
    "/kaggle/working/train_coco.json",
    "/kaggle/input/violence-data/violenceData/train/images"
)

register_coco_instances(
    "violence_val",
    {},
    "/kaggle/working/val_coco.json",
    "/kaggle/input/violence-data/violenceData/valid/images"
)

register_coco_instances(
    "violence_test",
    {},
    "/kaggle/working/test_coco.json",
    "/kaggle/input/violence-data/violenceData/test/images"
)

# Configuring the model used in this study 

cfg = get_cfg()

# VitDet base config
cfg.merge_from_file(
    "detectron2_repo/configs/ViTDet/standard/VIT_DETR_B_16_100ep.yaml"
)

cfg.DATASETS.TRAIN = ("violence_train",)
cfg.DATASETS.TEST  = ("violence_val",)
cfg.DATALOADER.NUM_WORKERS = 2

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 1e-5
cfg.SOLVER.MAX_ITER = 4000  # adjust as needed
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

cfg.OUTPUT_DIR = "/kaggle/working/vitdet_violence_results"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Training the model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

start_train = time.time()
trainer.train()
train_time = time.time() - start_train
print("Training completed in", train_time, "seconds")


evaluator = COCOEvaluator("violence_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "violence_val")

eval_results = inference_on_dataset(trainer.model, val_loader, evaluator)
print("Validation result:", eval_results)


# Evaluating on the test set

test_evaluator = COCOEvaluator("violence_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
test_loader = build_detection_test_loader(cfg, "violence_test")
test_results = inference_on_dataset(trainer.model, test_loader, test_evaluator)
print("Test result:", test_results)



#RObust evaluation using precision, recall and F1-score
params_M = sum(p.numel() for p in trainer.model.parameters())/1e6

process = psutil.Process(os.getpid())
mem_GB = process.memory_info().rss/(1024**3)
peak_mem_GB = torch.cuda.max_memory_allocated()/1024**3

dummy = torch.randn(1,3,800,800).to(cfg.MODEL.DEVICE)
flops, _ = profile(trainer.model, inputs=(dummy,), verbose=False)
gflops = flops/1e9

print("Params (M):", params_M)
print("Memory (GB):", mem_GB)
print("Peak Memory (GB):", peak_mem_GB)
print("GFLOPs:", gflops)

import time

trainer.model.eval()
start_inf = time.time()

with torch.no_grad():
    for images, _ in test_loader:
        images = list(img.to(cfg.MODEL.DEVICE) for img in images)
        _ = trainer.model(images)

inf_time = (time.time()-start_inf)/len(test_loader)*1000
throughput = len(test_dataset)/(time.time()-start_inf)

print("Inference time (ms):",inf_time)
print("Throughput (img/s):",throughput)

power_W = 250
inf_energy_kWh = (power_W * inf_time/1000)/3600
train_energy_kWh = (power_W * train_time)/3600
carbon_kg = train_energy_kWh * 0.475  # median factor

print("Inference Energy (kWh):",inf_energy_kWh)
print("Training Energy (kWh):",train_energy_kWh)
print("Carbon Emission (kg):",carbon_kg)

print("========== FINAL METRICS ==========")
print(f"Precision   : {test_results['bbox']['AP50']*100:.2f}%")
print(f"Recall      : {test_results['bbox']['AR']*100:.2f}%")
print(f"F1-score    : {(2*test_results['bbox']['AP50']*test_results['bbox']['AR']/(test_results['bbox']['AP50']+test_results['bbox']['AR']))*100:.2f}%")
print(f"mAP@0.5     : {test_results['bbox']['AP50']*100:.2f}%")
print(f"mAP@0.5:0.95: {test_results['bbox']['AP']*100:.2f}%")
print("Params (M)     :",params_M)
print("GFLOPs         :",gflops)
print("Inference Time :",inf_time,"ms")
print("Throughput     :",throughput,"img/s")
print("Memory Usage   :",mem_GB,"GB")
print("Peak Memory    :",peak_mem_GB,"GB")
print("Checkpoint Size:",os.path.getsize(cfg.OUTPUT_DIR+"/model_final.pth")/1e6,"MB")
print("Training Time  :",train_time,"s")
print("Inference Energy (kWh):",inf_energy_kWh)
print("Training Energy (kWh):",train_energy_kWh)
print("Carbon Emission (kg) :",carbon_kg)



