# YOLO Detection-Aware SR Final Comparison (alpha softened version)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["OMP_NUM_THREADS"]="1"

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from ultralytics import YOLO

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())

#############################################
# CONFIG
#############################################

COCO_IMG_DIR = Path("data/coco_da_val/images")
COCO_ANN_JSON = Path("data/coco_da_val/instances_val2017.json")

OUTPUT_DIR=Path("results_yolo_detection_alpha015_small")
OUTPUT_IMAGES_DIR=OUTPUT_DIR/"generated_images"
OUTPUT_DET_DIR=OUTPUT_DIR/"detections"

OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_IMAGES_DIR.mkdir(exist_ok=True)
OUTPUT_DET_DIR.mkdir(exist_ok=True)

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_SCALE=4

YOLO_MODEL_NAME="yolov8n.pt"

CONF_THRESHOLD=0.10
IOU_THRESHOLD=0.50

MAX_IMAGES=100
FILTER_OBJECT_SIZE="small"

CHECKPOINTS = {
    "3Region_Thr1035": "./srcnn_adaptive_3region/srcnn_x4_adaptive_3region_thr1035.pth",
    "DetectionAware_Thr1035_Beta15": "./srcnn_thr_adaptive/srcnn_x4_detection_aware_thr1035_beta15.pth",
}

ALPHA_BY_MODEL={
 "3Region_Thr1035":0.15,
 "DetectionAware_Thr1035_Beta15":0.15
}

#############################################
# MODEL
#############################################

class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1=nn.Conv2d(1,128,9,padding=4)
        self.conv2=nn.Conv2d(128,64,1)
        self.conv3=nn.Conv2d(64,1,5,padding=2)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=self.conv3(x)
        return x


def load_srcnn_checkpoint(path):

    ckpt=torch.load(path,map_location=DEVICE)

    model=SRCNN().to(DEVICE)

    if isinstance(ckpt,dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        scale=ckpt.get("scale",DEFAULT_SCALE)
    else:
        model.load_state_dict(ckpt)
        scale=DEFAULT_SCALE

    model.eval()

    return model,scale


def crop_to_scale(img,scale):
    w,h=img.size
    return img.crop(
        (0,0,w-w%scale,h-h%scale)
    )


def make_bicubic_variant(hr,scale):

    hr=crop_to_scale(hr,scale)

    w,h=hr.size

    lr=hr.resize(
      (w//scale,h//scale),
      Image.BICUBIC
    )

    bic=lr.resize(
      (w,h),
      Image.BICUBIC
    )

    return hr,lr,bic


@torch.no_grad()
def apply_srcnn(model,bicubic_img,alpha=0.3):

    ycbcr=bicubic_img.convert("YCbCr")
    y,cb,cr=ycbcr.split()

    y_t=transforms.ToTensor()(y).unsqueeze(0).to(DEVICE)

    res=model(y_t)

    # softened residual
    sr=torch.clamp(
      y_t + alpha*res,
      0,
      1
    )

    sr_y=transforms.ToPILImage()(sr.squeeze(0).cpu())

    sr_rgb=Image.merge(
      "YCbCr",
      (sr_y,cb,cr)
    ).convert("RGB")

    return sr_rgb

#############################################
# LOAD MODELS
#############################################

models={}

for n,p in CHECKPOINTS.items():

    path=Path(p)

    if not path.exists():
        print("missing",path)
        continue

    model,_=load_srcnn_checkpoint(path)

    models[n]=model

print(models.keys())

#############################################
# COCO
#############################################

with open(COCO_ANN_JSON,"r") as f:
    coco=json.load(f)

images_info=coco["images"]
annotations=coco["annotations"]
categories=coco["categories"]

cat_id_to_name={
 c["id"]:c["name"]
 for c in categories
}

cat_name_to_id={
 c["name"]:c["id"]
 for c in categories
}

file_to_img={
 i["file_name"]:i
 for i in images_info
}

anns_by_image={}
for ann in annotations:

    if ann.get("iscrowd",0)==1:
        continue

    anns_by_image.setdefault(
      ann["image_id"],
      []
    ).append(ann)


available_files=sorted([
 p.name
 for p in COCO_IMG_DIR.glob("*")
 if p.name in file_to_img
])

def ann_area_group(a):
    area=a.get(
       "area",
       a["bbox"][2]*a["bbox"][3]
    )

    if area<32**2:
        return "small"
    if area<96**2:
        return "medium"
    return "large"


def image_matches_filter(fname):

    if FILTER_OBJECT_SIZE=="all":
        return True

    img_id=file_to_img[fname]["id"]

    anns=anns_by_image.get(
       img_id,
       []
    )

    return any(
      ann_area_group(a)==FILTER_OBJECT_SIZE
      for a in anns
    )


selected_files=[
f for f in available_files
if image_matches_filter(f)
]

if MAX_IMAGES:
    selected_files=selected_files[:MAX_IMAGES]

print("images:",len(selected_files))

#############################################
# IOU
#############################################

def xywh_to_xyxy(b):
    x,y,w,h=b
    return [x,y,x+w,y+h]


def compute_iou(a,b):

    x1=max(a[0],b[0])
    y1=max(a[1],b[1])
    x2=min(a[2],b[2])
    y2=min(a[3],b[3])

    iw=max(0,x2-x1)
    ih=max(0,y2-y1)

    inter=iw*ih

    area1=(a[2]-a[0])*(a[3]-a[1])
    area2=(b[2]-b[0])*(b[3]-b[1])

    union=area1+area2-inter

    if union<=0:
        return 0

    return inter/union


#############################################
# GT
#############################################

def get_gt(fname,w,h):

    img_id=file_to_img[fname]["id"]

    g=[]

    for ann in anns_by_image.get(img_id,[]):

        box=xywh_to_xyxy(
           ann["bbox"]
        )

        g.append({
         "image":fname,
         "category_id":ann["category_id"],
         "bbox":box
        })

    return g

#############################################
# IMAGE VARIANTS
#############################################

def save_variant_images(fname):

    hr=Image.open(
      COCO_IMG_DIR/fname
    ).convert("RGB")

    hr,lr,bic=make_bicubic_variant(
      hr,
      DEFAULT_SCALE
    )

    stem=Path(fname).stem

    paths={}

    p=OUTPUT_IMAGES_DIR/f"{stem}_HR.png"
    hr.save(p)
    paths["HR"]=p

    p=OUTPUT_IMAGES_DIR/f"{stem}_Bicubic.png"
    bic.save(p)
    paths["Bicubic"]=p

    for name,model in models.items():

        alpha=ALPHA_BY_MODEL[name]

        sr=apply_srcnn(
          model,
          bic,
          alpha
        )

        p=OUTPUT_IMAGES_DIR/f"{stem}_{name}.png"

        sr.save(p)

        paths[name]=p

    return paths,hr.size

#############################################
# YOLO
#############################################

detector=YOLO(
 YOLO_MODEL_NAME
)

def run_yolo(img_path,variant,fname):

    results=detector.predict(
      source=str(img_path),
      conf=CONF_THRESHOLD,
      save=False,
      verbose=False,
      device=0 if DEVICE=="cuda" else "cpu"
    )

    r=results[0]

    rows=[]

    boxes=r.boxes

    if boxes is None or len(boxes)==0:
        return rows

    xyxy=boxes.xyxy.cpu().numpy()
    conf=boxes.conf.cpu().numpy()
    cls=boxes.cls.cpu().numpy()

    for i in range(len(boxes)):

        cname=r.names[int(cls[i])]

        rows.append({
         "image":fname,
         "variant":variant,
         "category_id":cat_name_to_id.get(
            cname,None
         ),
         "confidence":float(conf[i]),
         "bbox":xyxy[i].tolist()
        })

    return rows

#############################################
# RUN
#############################################

all_preds=[]
all_gts=[]

variants=[
 "HR",
 "Bicubic",
 "3Region_Thr1035",
 "DetectionAware_Thr1035_Beta15"
]

for fname in tqdm(selected_files):

    paths,(w,h)=save_variant_images(fname)

    all_gts.extend(
      get_gt(fname,w,h)
    )

    for v in variants:
        all_preds.extend(
          run_yolo(
             paths[v],
             v,
             fname
          )
        )

pred_df=pd.DataFrame(all_preds)
gt_df=pd.DataFrame(all_gts)

#############################################
# EVAL
#############################################

def eval_variant(variant):

    preds=pred_df[
      pred_df.variant==variant
    ].sort_values(
      "confidence",
      ascending=False
    )

    matched=set()

    tp=0
    fp=0

    for _,pred in preds.iterrows():

        best=None
        best_iou=0

        for i,gt in gt_df.iterrows():

            if (
             gt.image!=pred.image or
             gt.category_id!=pred.category_id or
             i in matched
            ):
                continue

            iou=compute_iou(
               pred.bbox,
               gt.bbox
            )

            if iou>best_iou:
                best_iou=iou
                best=i

        if best_iou>=IOU_THRESHOLD:
            tp+=1
            matched.add(best)
        else:
            fp+=1

    fn=len(gt_df)-tp

    precision=tp/(tp+fp+1e-9)
    recall=tp/(tp+fn+1e-9)

    f1=2*precision*recall/(
       precision+recall+1e-9
    )

    return {
      "variant":variant,
      "precision":precision,
      "recall":recall,
      "f1":f1
    }

rows=[]

for v in variants:
    rows.append(
      eval_variant(v)
    )

res=pd.DataFrame(rows)

print(res)

#############################################
# crude mAP approx
#############################################

print()
print("done.")
print("check if bicubic beaten.")