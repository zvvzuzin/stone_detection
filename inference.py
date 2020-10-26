import mmcv
from mmcv import Config
from mmdet.apis import inference_detector, init_detector
import numpy as np

class Inference_model:
    def __init__(self, path_to_cfg = 'configs/custom_maskrcnn.py', path_to_model = 'weights.pth'):
        cfg = Config.fromfile('configs/custom_maskrcnn.py')
        self.model = init_detector(cfg, path_to_model, device='cpu')
        self.model.CLASSES = ('stone', ) # UserWarning: Class names are not saved in the checkpoint's meta data, use COCO classes by default.
        self.model.classes = self.model.CLASSES
        
    def predict(self, img):
        result = inference_detector(self.model, img)
        return result[0][0], result[1][0]