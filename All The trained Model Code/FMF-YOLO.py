
import subprocess
subprocess.check_call(['pip', 'install', 'ultralytics'])

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, Bottleneck, SPPF

# Feature Enhancement Module (FEM)
class FEM(nn.Module):
    def __init__(self, c1, c2, dilation_rates=[1, 2, 3, 4]):
        super(FEM, self).__init__()
        self.conv1 = Conv(c1, c2, k=1)
        
        self.atrous_convs = nn.ModuleList()
        for rate in dilation_rates:
            self.atrous_convs.append(
                Conv(c2, c2, k=3, p=rate, d=rate, act=True)
            )

        self.rfb_conv = Conv(c2 * len(dilation_rates), c2, k=1, act=True)
        
    def forward(self, x):
        x = self.conv1(x)

        atrous_outputs = []
        for atrous_conv in self.atrous_convs:
            atrous_outputs.append(atrous_conv(x))
        
        x = torch.cat(atrous_outputs, dim=1)
        x = self.rfb_conv(x)
        
        return x

# Custom Fusion Module for BiFPN
class CustomFusion(nn.Module):
    def __init__(self, c1, c2):
        super(CustomFusion, self).__init__()
        self.csp = C2f(c1, c2, n=1, shortcut=False)
        self.cbs = Conv(c2, c2, k=1, act=True)
        
    def forward(self, x):
        x = self.csp(x)
        x = self.cbs(x)
        return x

# Feature Fusion Module (FFM) with BiFPN architecture
class FFM(nn.Module):
    def __init__(self, channels_list, c2):
        super(FFM, self).__init__()
        self.channels_list = channels_list
        self.crc_conv = nn.ModuleList()
        for c1 in channels_list:
            self.crc_conv.append(Conv(c1, c2, k=1, act=True))

        self.fusion_blocks = nn.ModuleList()
        for _ in range(len(channels_list) - 1):
            self.fusion_blocks.append(CustomFusion(c2 * 2, c2))
        
        self.weights = nn.Parameter(torch.ones(len(channels_list)))
        
    def forward(self, features):
   
        crc_features = []
        for i, (feature, conv) in enumerate(zip(features, self.crc_conv)):
            crc_features.append(conv(feature) * self.weights[i])
        
 
        fused = crc_features[0]
        for i in range(1, len(crc_features)):
            # Resize to match dimensions
            if fused.shape[2:] != crc_features[i].shape[2:]:
                target_size = fused.shape[2:]
                resized_feature = F.interpolate(crc_features[i], size=target_size, mode='nearest')
            else:
                resized_feature = crc_features[i]
            
    
            combined = torch.cat([fused, resized_feature], dim=1)
            fused = self.fusion_blocks[i-1](combined)
        
        return fused

# Create proper YAML configuration 
def create_custom_yaml():
    yaml_content = """
# Modified YOLOv9m with FEM and FFM
nc: 2  # number of classes
names: ['violence', 'non-violence']  # class names

# YOLOv9m backbone with FEM
backbone:
  #- [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  #- [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  #- [-1, 1, FEM, [256]]  # 2-FEM Module added
  #- [-1, 6, C2f, [256, True]]
  #- [-1, 1, Conv, [512, 3, 2]]  # 4-P3/8
  #- [-1, 6, C2f, [512, True]]
  #- [-1, 1, Conv, [1024, 3, 2]]  # 6-P4/16
  #- [-1, 6, C2f, [1024, True]]
  #- [-1, 1, Conv, [1024, 3, 2]]  # 8-P5/32
  #- [-1, 6, C2f, [1024, True]]
  #- [-1, 1, SPPF, [1024, 5]]

# YOLOv9m head with FFM
head:
  #- [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  #- [[-1, 7], 1, Concat, [1]]  # cat backbone P4
  #- [-1, 3, C2f, [512]]  # 12
  #- [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  #- [[-1, 5], 1, Concat, [1]]  # cat backbone P3
  #- [-1, 3, C2f, [256]]  # 15 (P3/8-small)
  #- [-1, 1, Conv, [256, 3, 2]]
  #- [[-1, 12], 1, Concat, [1]]  # cat head P4
  #- [-1, 3, C2f, [512]]  # 18 (P4/16-medium)
  #- [-1, 1, Conv, [512, 3, 2]]
  #- [[-1, 9], 1, Concat, [1]]  # cat head P5
  #- [-1, 3, C2f, [1024]]  # 21 (P5/32-large)
  #- [[15, 18, 21], 1, FFM, [256]]  # FFM Module for multi-scale fusion
  #- [[0], 1, Detect, [nc]]  # Detect(P3, P4, P5)
"""
    
    with open('/kaggle/working/custom_yolov9m.yaml', 'w') as f:
        f.write(yaml_content)
    return '/kaggle/working/custom_yolov9m.yaml'


def train_yolov9m_kaggle():
    """
    Simplified training function optimized for Kaggle environment
    """
  
    data_yaml_path = '/kaggle/input/violence-data/violenceData/data.yaml'
    
    # Verify dataset exists
    if not os.path.exists(data_yaml_path):
        print(" Data.yaml not found at specified path")
        print("Available files in violence-data:")
        import subprocess
        result = subprocess.run(['find', '/kaggle/input', '-name', '*.yaml', '-type', 'f'], 
                              capture_output=True, text=True)
        print("Found YAML files:", result.stdout)
        return None
    
 
    try:
        model = YOLO('yolov9m.pt')
        print(" YOLOv9m model loaded successfully")
    except:
    
        print(" YOLOv9m not available, trying YOLOv9s")
        model = YOLO('yolov9s.pt')
    
    # Display model information
    print(f" Model: {model.__class__.__name__}")
    print(f" Dataset config: {data_yaml_path}")
    

    try:
        with open(data_yaml_path, 'r') as f:
            data_config = f.read()
            print(" Data configuration:")
            print(data_config)
    except Exception as e:
        print(f" Error reading data.yaml: {e}")
        return None
    

    training_config = {
        'data': data_yaml_path,
        'epochs': 60,
        'batch': 16,
        'imgsz': 640,
        'patience': 10,
        'save': True,
        'exist_ok': True,
        'project': '/kaggle/working/training_results',
        'name': 'yolov9m_violence_detection',
        'device': 0 if torch.cuda.is_available() else 'cpu', 
        'workers': 2,  
        'optimizer': 'auto',
        'momentum': 0.937,
     
    }
    
    print(" Starting training with configuration:")
    for key, value in training_config.items():
        print(f"   {key}: {value}")
    
    try:
        results = model.train(**training_config)
        print(" Training completed successfully!")
        return model, results
    except Exception as e:
        print(f" Training failed: {e}")
        return None, None

# Model evaluation function
def evaluate_model(model_path):
    """
    Evaluate the trained model
    """
    try:
        model = YOLO(model_path)
        
        # Validation metrics
        metrics = model.val(
            data='/kaggle/input/violence-data/violenceData/data.yaml',
            split='val',
            imgsz=640,
            batch=16,
            save_json=True,
            save_conf=True
        )
        
        return metrics
    except Exception as e:
        print(f" Evaluation failed: {e}")
        return None

# Main execution block for Kaggle
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 VIOLENCE DETECTION MODEL TRAINING - KAGGLE VERSION")
    print("=" * 60)
    

    print("🔍 Environment check:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
 
    print("\n Dataset verification:")
    dataset_paths = [
        '/kaggle/input/violence-data/violenceData/train/images',
        '/kaggle/input/violence-data/violenceData/valid/images',
        '/kaggle/input/violence-data/violenceData/data.yaml'
    ]
    
    for path in dataset_paths:
        exists = os.path.exists(path)
        status =  if exists else
        print(f"   {status} {path}")
        
        if exists and 'images' in path:
            # Count images
            try:
                image_count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"     Images found: {image_count}")
            except:
                pass
    
    # Create output directory
    os.makedirs('/kaggle/working/training_results', exist_ok=True)
    
    # Start training
    print("\n Starting model training...")
    model, results = train_yolov9m_kaggle()
    
    if model is not None:
        # Save the best model
        best_model_path = '/kaggle/working/training_results/yolov9m_violence_detection/weights/best.pt'
        
        if os.path.exists(best_model_path):
            print(f"\nBest model saved at: {best_model_path}")
            
       
            print("\n Evaluating model performance...")
            metrics = evaluate_model(best_model_path)
            
            if metrics:
                print(" Evaluation completed!")
            
                if hasattr(metrics, 'box'):
                    print(f"   mAP50: {metrics.box.map50:.4f}")
                    print(f"   mAP50-95: {metrics.box.map:.4f}")
                    print(f"   Precision: {metrics.box.precision:.4f}")
                    print(f"   Recall: {metrics.box.recall:.4f}")
        
    
        try:
            # Create results summary
            summary = {
                'model': 'Modified YOLOv9m with FEM and FFM',
                'dataset': 'Violence Detection',
                'classes': ['violence', 'non-violence'],
                'training_epochs': 60,
                'batch_size': 16,
                'input_size': 640,
                'best_model_path': best_model_path if os.path.exists(best_model_path) else 'Not found'
            }
            
   
            import json
            with open('/kaggle/working/training_summary.json', 'w') as f:
                json.dump(summary, f, indent=4)
            
        
            
        except Exception as e:
          
    print("\n" + "=" * 60)
    print(" Training process completed!")
    print("=" * 60)

    print("\n Final working directory structure:")
    result = subprocess.run(['find', '/kaggle/working', '-type', 'f', '-name', '*.pt', '-o', '-name', '*.json', '-o', '-name', '*.yaml'], 
                          capture_output=True, text=True)
    print(result.stdout)