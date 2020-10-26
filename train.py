import os
import mmcv
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

# path_to_mmdet = '/home/vasily/proj/mmdetection'
# path_to_dataset = 'asbest/stones_detection'

cfg = Config.fromfile('configs/custom_maskrcnn.py')

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# cfg.log_config.interval = 2

# Set seed thus the results are more reproducible
# cfg.seed = 0
set_random_seed(0, deterministic=False) # ?

cfg.total_epochs = 100
# Build the detector
model = build_detector(cfg.model,
                       train_cfg=cfg.train_cfg,
                       test_cfg=cfg.test_cfg)

# train
# Create work_dir
mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=False)