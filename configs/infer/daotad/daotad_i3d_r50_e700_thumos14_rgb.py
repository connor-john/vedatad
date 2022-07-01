# 1. data
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
num_frames = 768
img_shape = (112, 112)
overlap_ratio = 0.25

data_pipeline=[
    dict(
        typename='OverlapCropAug',
        num_frames=num_frames,
        overlap_ratio=overlap_ratio,
        transforms=[
            dict(typename='TemporalCrop'),
            dict(typename='LoadFrames', to_float32=True),
            dict(typename='SpatialCenterCrop', crop_size=img_shape),
            dict(typename='Normalize', **img_norm_cfg),
            dict(typename='Pad', size=(num_frames, *img_shape)),
            dict(typename='DefaultFormatBundle'),
            dict(typename='Collect', keys=['imgs'])
    ])
]

# 2. model
num_classes = 20
strides = [8, 16, 32, 64, 128]
use_sigmoid = True
scales_per_octave = 5
octave_base_scale = 2
num_anchors = scales_per_octave

model = dict(
    typename='SingleStageDetector',
    backbone=dict(
        typename='ResNet3d',
        depth=50,
        norm_eval=True,
        out_indices=(3, ),
        frozen_stages=1,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False),
    neck=[
        dict(
            typename='SRM',
            srm_cfg=dict(
                typename='AdaptiveAvgPool3d', output_size=(None, 1, 1))),
        dict(
            typename='TDM',
            in_channels=2048,
            stage_layers=(1, 1, 1, 1),
            out_channels=512,
            conv_cfg=dict(typename='Conv1d'),
            norm_cfg=dict(typename='SyncBN'),
            act_cfg=dict(typename='ReLU'),
            out_indices=(0, 1, 2, 3, 4)),
        dict(
            typename='FPN',
            in_channels=[2048, 512, 512, 512, 512],
            out_channels=256,
            num_outs=5,
            start_level=0,
            conv_cfg=dict(typename='Conv1d'),
            norm_cfg=dict(typename='SyncBN'))
    ],
    head=dict(
        typename='RetinaHead',
        num_classes=num_classes,
        num_anchors=num_anchors,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        use_sigmoid=use_sigmoid,
        conv_cfg=dict(typename='Conv1d'),
        norm_cfg=dict(typename='SyncBN'),
    ))

# 3. engine
meshgrid = dict(
    typename='SegmentAnchorMeshGrid',
    strides=strides,
    base_anchor=dict(
        typename='SegmentBaseAnchor',
        base_sizes=strides,
        octave_base_scale=octave_base_scale,
        scales_per_octave=scales_per_octave))

segment_coder = dict(
    typename='DeltaSegmentCoder',
    target_means=[.0, .0],
    target_stds=[1.0, 1.0])

# 3.2 infer engine
infer_engine = dict(
    typename='ValEngine',
    model=model,
    meshgrid=meshgrid,
    converter=dict(
        typename='SegmentAnchorConverter',
        num_classes=num_classes,
        segment_coder=segment_coder,
        nms_pre=1000,
        use_sigmoid=use_sigmoid),
    num_classes=num_classes,
    test_cfg=dict(
        score_thr=0.005,
        nms=dict(typename='nmw', iou_thr=0.5),
        max_per_video=1200),
    use_sigmoid=use_sigmoid)

# 4. checkpoint
weights = dict(
    filepath='workdir/daotad_i3d_r50_e700_thumos14_rgb/epoch_1200_weights.pth'
)

# 7. misc
dist_params = dict(backend='nccl')
log_level = 'INFO'
find_unused_parameters = True
deterministic = True
