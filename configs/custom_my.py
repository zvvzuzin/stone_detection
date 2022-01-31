_base_ = '/home/vasily/proj/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'

classes = ["stone"]
CLASSES = classes

# learning policy

num_classes = 1

# '../_base_/models/mask_rcnn_r50_fpn.py'
# model settings
model = dict(
    type='MaskRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        in_channels=1,),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))
# model = dict(
#     type='MaskRCNN',
#     pretrained='torchvision://resnet50',
#     backbone=dict(
#         type='ResNet',
#         depth=50,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         norm_eval=True,
#         style='pytorch'),
#     neck=dict(...),
#     rpn_head=dict(...),
#     roi_head=dict(...))

# model = dict(
#     type='MaskRCNN',
#     pretrained='torchvision://resnet50',
#     backbone=dict(
#         type='ResNet',
#         in_channels=1,
#         depth=50,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         norm_eval=False,
#         style='pytorch'),
#     neck=dict(
#         type='FPN',
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         num_outs=5),
#     rpn_head=dict(
#         type='RPNHead',
#         in_channels=256,
#         feat_channels=256,
#         anchor_generator=dict(
#             type='AnchorGenerator',
#             scales=[8],
#             ratios=[0.5, 1.0, 2.0],
#             strides=[4, 8, 16, 32, 64]),
#         bbox_coder=dict(
#             type='DeltaXYWHBBoxCoder',
#             target_means=[.0, .0, .0, .0],
#             target_stds=[1.0, 1.0, 1.0, 1.0]),
#         loss_cls=dict(
#             type='CrossEntropyLoss', use_sigmoid=True, loss_weight=3.0),
#         loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
#     roi_head=dict(
#         type='StandardRoIHead',
#         bbox_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#         bbox_head=dict(
#             type='Shared2FCBBoxHead',
#             in_channels=256,
#             fc_out_channels=1024,
#             roi_feat_size=7,
#             num_classes=num_classes,
#             bbox_coder=dict(
#                 type='DeltaXYWHBBoxCoder',
#                 target_means=[0., 0., 0., 0.],
#                 target_stds=[0.1, 0.1, 0.2, 0.2]),
#             reg_class_agnostic=False,
#             loss_cls=dict(
#                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#             loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
#         mask_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#         mask_head=dict(
#             type='FCNMaskHead',
#             num_convs=4,
#             in_channels=256,
#             conv_out_channels=256,
#             num_classes=num_classes,
#             loss_mask=dict(
#                 type='CrossEntropyLoss', use_mask=True, loss_weight=1))))

    # model training and testing settings
train_cfg = dict(  # Config of training hyperparameters for rpn and rcnn
    rpn=dict(  # Training config of rpn
        assigner=dict(  # Config of assigner
            type='MaxIoUAssigner',  # Type of assigner, MaxIoUAssigner is used for many common detectors. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/max_iou_assigner.py#L10 for more details.
            pos_iou_thr=0.7,  # IoU >= threshold 0.7 will be taken as positive samples
            neg_iou_thr=0.3,  # IoU < threshold 0.3 will be taken as negative samples
            min_pos_iou=0.3,  # The minimal IoU threshold to take boxes as positive samples
            match_low_quality=True,  # Whether to match the boxes under low quality (see API doc for more details).
            ignore_iof_thr=-1),  # IoF threshold for ignoring bboxes
        sampler=dict(  # Config of positive/negative sampler
            type='RandomSampler',  # Type of sampler, PseudoSampler and other samplers are also supported. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/samplers/random_sampler.py#L8 for implementation details.
            num=256,  # Number of samples
            pos_fraction=0.5,  # The ratio of positive samples in the total samples.
            neg_pos_ub=-1,  # The upper bound of negative samples based on the number of positive samples.
            add_gt_as_proposals=False),  # Whether add GT as proposals after sampling.
        allowed_border=-1,  # The border allowed after padding for valid anchors.
        pos_weight=-1,  # The weight of positive samples during training.
        debug=False),  # Whether to set the debug mode
    rpn_proposal=dict(  # The config to generate proposals during training
        nms_across_levels=False,  # Whether to do NMS for boxes across levels. Only work in `GARPNHead`, naive rpn does not support do nms cross levels.
        nms_pre=2000,  # The number of boxes before NMS
        nms_post=1000,  # The number of boxes to be kept by NMS, Only work in `GARPNHead`.
        max_per_img=1000,  # The number of boxes to be kept after NMS.
        nms=dict( # Config of NMS
            type='nms',  # Type of NMS
            iou_threshold=0.7 # NMS threshold
            ),
        min_bbox_size=0),  # The allowed minimal box size
    rcnn=dict(  # The config for the roi heads.
        assigner=dict(  # Config of assigner for second stage, this is different for that in rpn
            type='MaxIoUAssigner',  # Type of assigner, MaxIoUAssigner is used for all roi_heads for now. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/max_iou_assigner.py#L10 for more details.
            pos_iou_thr=0.5,  # IoU >= threshold 0.5 will be taken as positive samples
            neg_iou_thr=0.5,  # IoU < threshold 0.5 will be taken as negative samples
            min_pos_iou=0.5,  # The minimal IoU threshold to take boxes as positive samples
            match_low_quality=False,  # Whether to match the boxes under low quality (see API doc for more details).
            ignore_iof_thr=-1),  # IoF threshold for ignoring bboxes
        sampler=dict(
            type='RandomSampler',  # Type of sampler, PseudoSampler and other samplers are also supported. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/samplers/random_sampler.py#L8 for implementation details.
            num=512,  # Number of samples
            pos_fraction=0.25,  # The ratio of positive samples in the total samples.
            neg_pos_ub=-1,  # The upper bound of negative samples based on the number of positive samples.
            add_gt_as_proposals=True
        ),  # Whether add GT as proposals after sampling.
        mask_size=28,  # Size of mask
        pos_weight=-1,  # The weight of positive samples during training.
        debug=False))  # Whether to set the debug mode


test_cfg = dict(  # Config for testing hyperparameters for rpn and rcnn
    rpn=dict(  # The config to generate proposals during testing
        nms_across_levels=False,  # Whether to do NMS for boxes across levels. Only work in `GARPNHead`, naive rpn does not support do nms cross levels.
        nms_pre=1000,  # The number of boxes before NMS
        nms_post=1000,  # The number of boxes to be kept by NMS, Only work in `GARPNHead`.
        max_per_img=1000,  # The number of boxes to be kept after NMS.
        nms=dict( # Config of NMS
            type='nms',  #Type of NMS
            iou_threshold=0.7 # NMS threshold
            ),
        min_bbox_size=0),  # The allowed minimal box size
    rcnn=dict(  # The config for the roi heads.
        score_thr=0.05,  # Threshold to filter out boxes
        nms=dict(  # Config of NMS in the second stage
            type='nms',  # Type of NMS
            iou_threshold=0.3),  # NMS threshold
        max_per_img=100,  # Max number of detections of each image
        mask_thr_binary=0.5))  # Threshold of mask prediction

dataset_type = 'CocoDataset'
# data_root_pits_300920 = '/home/vasily/datasets/asbestos/pits/300920'
# data_root_pits_161120 = '/home/vasily/datasets/asbestos/pits/161120'
# data_root_pits_161220 = '/home/vasily/datasets/asbestos/pits/161220'
data_root_transporter = '/home/vasily/datasets/asbest_old/tr_stones/'
# dataset_type = 'StonesDataset'
# data_root_common = '/home/vasily/datasets/asbest/pits/'
# data_root_small_pits = '/home/vasily/datasets/asbest/camera_pits/'

# data_root_shelves = '/home/vasily/datasets/asbest/stones_on_shelves/'

# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg = dict(mean=[123], std=[58], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1600, 1200), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='RandomCrop', crop_size=(1333, 800)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale', to_float32=True),
    dict(type='MultiScaleFlipAug',
        img_scale=(1600, 1200),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type="DefaultFormatBundle"),
             dict(type='Collect', keys=['img']),
         ])
]
dataset_transporter = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root_transporter + 'annotation/annotation.json',
        img_prefix=data_root_transporter + 'images/',
        pipeline=train_pipeline,
        classes=classes))

dataset_pits_300920 = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type=dataset_type,
        ann_file='/home/vasily/datasets/asbestos/pits/300920/annotation_300920.json',
        img_prefix='/home/vasily/datasets',
        pipeline=train_pipeline,
        classes=classes))

dataset_pits_161120 = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type=dataset_type,
        ann_file='/home/vasily/datasets/asbestos/pits/161120/annotation_161120.json',
        img_prefix='/home/vasily/datasets',
        pipeline=train_pipeline,
        classes=classes))

dataset_pits_161220 = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type=dataset_type,
        ann_file='/home/vasily/datasets/asbestos/pits/161220/annotation_161220.json',
        img_prefix='/home/vasily/datasets',
        pipeline=train_pipeline,
        classes=classes))

dataset_pits_020221 = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type=dataset_type,
        ann_file='/home/vasily/datasets/asbestos/pits/020221/annotation_020221.json',
        img_prefix='/home/vasily/datasets',
        pipeline=train_pipeline,
        classes=classes))

dataset_pits_111121 = dict(
    type='RepeatDataset',
    times=2,
    dataset=dict(
        type=dataset_type,
        ann_file='/home/vasily/datasets/asbestos/pits/111121/annotation_111121.json',
        img_prefix='/home/vasily/datasets/asbestos/pits/111121',
        pipeline=train_pipeline,
        classes=classes))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
#     train = [dataset_transporter],
    train=[dataset_pits_300920, dataset_pits_161120, dataset_pits_020221, dataset_pits_111121],
#     train=[dataset_pits_111121],
    val=dict(
        type=dataset_type,
        ann_file='/home/vasily/datasets/asbestos/pits/111121/annotation_111121.json',
        img_prefix='/home/vasily/datasets/asbestos/pits/111121',
        pipeline=test_pipeline,
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file='/home/vasily/datasets/asbestos/pits/111121/annotation_111121.json',
        img_prefix='/home/vasily/datasets/asbestos/pits/111121',
        pipeline=test_pipeline,
        classes=classes))

evaluation = dict(  # The config to build the evaluation hook, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7 for more details.
    interval=1,  # Evaluation interval
    metric=['bbox', 'segm'])  # Metrics used during evaluation
# optimizer = dict(  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch
#     type='SGD',  # Type of optimizers, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/optimizer/default_constructor.py#L13 for more details
#     lr=0.02,  # Learning rate of optimizers, see detail usages of the parameters in the documentation of PyTorch
#     momentum=0.9,  # Momentum
#     weight_decay=0.0001)  # Weight decay of SGD
# optimizer_config = dict(  # Config used to build the optimizer hook, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8 for implementation details.
#     grad_clip=None)  # Most of the methods do not use gradient clip
# lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
#     policy='step',  # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
#     warmup='linear',  # The warmup policy, also support `exp` and `constant`.
#     warmup_iters=500,  # The number of iterations for warmup
#     warmup_ratio=
#     0.001,  # The ratio of the starting learning rate used for warmup
#     step=[8, 11])  # Steps to decay the learning rate
runner = dict(
    type='EpochBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_epochs=50) # Runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`
# evaluation = dict(metric=['bbox', 'segm'])

# '../_base_/default_runtime.py'
checkpoint_config = dict(interval=10)
# evaluation = dict(interval=5)

# yapf:disable
log_config = dict(
    interval=10,  # 50
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

gpu_ids = range(1)
work_dir = './checkpoints'
seed = 42