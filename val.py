import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('best.pt') # 选择训练好的权重路径
    model.val(data='VisDrone/VisDrone.yaml',
              split='val',
              imgsz=640,
              batch=32,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )