!pip install torch torchvision transformers timm pycocotools torchinfo thop codecarbon --quiet


import os
import time
import torch
import psutil
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import DetrImageProcessor, DetrForObjectDetection
from pycocotools.coco import COCO
from codecarbon import EmissionsTracker
from thop import profile
from sklearn.metrics import precision_score, recall_score, f1_score


class ViolenceDatasetYOLO(Dataset):
    def __init__(self, images_dir, labels_dir, processor, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.processor = processor
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.png','jpeg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, image_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        boxes = []
        labels = []
        label_txt = os.path.join(self.labels_dir, image_name.replace('.jpg','.txt').replace('.png','.txt'))
        
        if os.path.exists(label_txt):
            with open(label_txt,'r') as f:
                lines = f.readlines()
            w,h = Image.open(os.path.join(self.images_dir, image_name)).size
            for line in lines:
                cls,x_center,y_center,w_rel,h_rel = map(float, line.strip().split())
                bbox_w,bbox_h = w_rel*w, h_rel*h
                x0 = (x_center*w) - bbox_w/2
                y0 = (y_center*h) - bbox_h/2
                x1 = x0 + bbox_w
                y1 = y0 + bbox_h
                boxes.append([x0,y0,x1,y1])
                labels.append(int(cls))
        
        target = {"boxes": torch.tensor(boxes,dtype=torch.float32),
                  "labels": torch.tensor(labels,dtype=torch.long)}
        return image, target
    


processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
transform = transforms.Compose([
    transforms.Resize((800,800)),
    transforms.ToTensor()
])



train_dataset = ViolenceDatasetYOLO(
    "/kaggle/input/violence-data/violenceData/train/images",
    "/kaggle/input/violence-data/violenceData/train/labels",
    processor, transform
)

val_dataset = ViolenceDatasetYOLO(
    "/kaggle/input/violence-data/violenceData/valid/images",
    "/kaggle/input/violence-data/violenceData/valid/labels",
    processor, transform
)

test_dataset = ViolenceDatasetYOLO(
    "/kaggle/input/violence-data/violenceData/test/images",
    "/kaggle/input/violence-data/violenceData/test/labels",
    processor, transform
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=2)
test_loader  = DataLoader(test_dataset, batch_size=2)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=2,
    ignore_mismatched_sizes=True
).to(device)



optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

tracker = EmissionsTracker()
tracker.start()

start_time = time.time()

model.train()
for epoch in range(5):  # adjust as needed
    running_loss = 0
    for images, targets in train_loader:
        imgs = list(img.to(device) for img in images)
        targs = [{k:v.to(device) for k,v in t.items()} for t in targets]
        
        outputs = model(imgs, labels=targs)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")
    
training_time = time.time() - start_time
training_emissions = tracker.stop()
print("Training done!")


checkpoint_path = "/kaggle/working/rtdetr_violence.pt"
torch.save(model.state_dict(),checkpoint_path)

checkpoint_size = os.path.getsize(checkpoint_path)/(1024*1024)
print("Checkpoint saved:", checkpoint_size,"MB")


num_params = sum(p.numel() for p in model.parameters())/1e6
model_size_MB = num_params*4/1024  # approx

process = psutil.Process(os.getpid())
memory_usage_GB = process.memory_info().rss/(1024**3)
peak_memory_GB = torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else memory_usage_GB

dummy_input = torch.randn(1,3,800,800).to(device)
flops, params = profile(model, inputs=(dummy_input,), verbose=False)
gflops = flops/1e9

print("Params (M):", num_params)
print("Memory (GB):", memory_usage_GB)
print("Peak Mem (GB):", peak_memory_GB)
print("GFLOPs:", gflops)
all_preds = []
all_labels = []

model.eval()
start_inf = time.time()

with torch.no_grad():
    for images, targets in test_loader:
        imgs = list(img.to(device) for img in images)
        targets_cpu = [t["labels"].cpu().numpy() for t in targets]
        
        outputs = model(imgs)
        logits = outputs.logits
        preds = torch.argmax(logits,dim=-1).cpu().numpy()
        
        all_preds.extend(preds.flatten())
        all_labels.extend(np.concatenate(targets_cpu).flatten())

inf_time = (time.time()-start_inf)/len(test_loader)*1000
throughput = len(test_dataset)/((time.time()-start_inf))

precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)

print("Inference time (ms):",inf_time)
print("Throughput:", throughput)
print("Precision:",precision)
print("Recall:",recall)
print("F1-score:",f1)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def build_coco_results(model, loader):
    results = []
    idx = 0
    for images, targets in loader:
        imgs = list(img.to(device) for img in images)
        outputs = model(imgs).to("cpu")
        boxes = outputs.pred_boxes.tolist()
        scores = outputs.pred_scores.tolist()
        
        for i, img_path in enumerate(loader.dataset.image_files[idx:idx+len(images)]):
            img_id = idx + i
            for b,s in zip(boxes[i],scores[i]):
                results.append({
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [b[0],b[1],b[2]-b[0],b[3]-b[1]],
                    "score": s
                })
        idx += len(images)
    return results

gt_json = "/kaggle/input/violence-data/violenceData/test/labels/coco_test.json"
coco_gt = COCO(gt_json)
coco_dt = coco_gt.loadRes(build_coco_results(model,test_loader))

coco_eval = COCOeval(coco_gt,coco_dt,'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()


print("======== SUMMARY ========")
print("Model Size (MB):", model_size_MB)
print("Memory Usage (GB):", memory_usage_GB)
print("HSD (storage) Checkpoint Size (MB):", checkpoint_size)
print("Execution Time (Training) :", training_time, "s")
print("Peak Memory (GB):", peak_memory_GB)
print("Params (M):", num_params)
print("GFLOPs:",gflops)
print("Inference Time (ms):", inf_time)
print("Throughput (img/sec):", throughput)
print("Inference Energy (kWh): approx", (250*(inf_time/1000))/3600)
print("Carbon Emission (kg): approx", training_emissions)
print("\n--- EVALUATION ---")
print("Precision:",precision)
print("Recall:",recall)
print("F1-score:",f1)
print("mAP@0.5 and mAP0.5:0.95 from COCOEval above.")


