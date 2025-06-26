point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuSceneOcc'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadOccGTFromFile',
        data_root='data/nuscenes/',
        pc_range=[-40, -40, -1.0, 40, 40, 5.4]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-40, -40, -1.0, 40, 40, 5.4]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='CustomResizeCropFlipImage',
        data_aug_conf=dict(
            resize_lim=(0.386, 0.55),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True),
        training=True),
    dict(
        type='CustomGlobalRotScaleTransImage',
        flip_hv_ratio=[0.5, 0.5],
        pc_range=[-40, -40, -1.0, 40, 40, 5.4]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=[
            'img', 'voxel_semantics', 'mask_lidar', 'mask_camera',
            'prev_exists'
        ],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'scene_token', 'can_bus',
                   'ego2lidar', 'prev_idx', 'next_idx', 'ego2global',
                   'timestamp', 'img_trans_dict', 'ego_trans_dict',
                   'cam_intrinsic', 'cam2ego', 'pixel_wise_label'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccGTFromFile', data_root='data/nuscenes/'),
    dict(
        type='CustomResizeCropFlipImage',
        data_aug_conf=dict(
            resize_lim=(0.386, 0.55),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True),
        training=False),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                           'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                           'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                           'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                           'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                           'pcd_rotation', 'pts_filename',
                           'transformation_3d_flow', 'scene_token', 'can_bus',
                           'ego2lidar', 'prev_idx', 'next_idx', 'ego2global',
                           'timestamp', 'img_trans_dict', 'ego_trans_dict'))
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='NuSceneOcc',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/occ_infos_temporal_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadOccGTFromFile',
                data_root='data/nuscenes/',
                pc_range=[-40, -40, -1.0, 40, 40, 5.4]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=False),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-40, -40, -1.0, 40, 40, 5.4]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='CustomResizeCropFlipImage',
                data_aug_conf=dict(
                    resize_lim=(0.386, 0.55),
                    final_dim=(256, 704),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=True),
            dict(
                type='CustomGlobalRotScaleTransImage',
                flip_hv_ratio=[0.5, 0.5],
                pc_range=[-40, -40, -1.0, 40, 40, 5.4]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='Collect3D',
                keys=[
                    'img', 'voxel_semantics', 'mask_lidar', 'mask_camera',
                    'prev_exists'
                ],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                           'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                           'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                           'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                           'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                           'pcd_rotation', 'pts_filename',
                           'transformation_3d_flow', 'scene_token', 'can_bus',
                           'ego2lidar', 'prev_idx', 'next_idx', 'ego2global',
                           'timestamp', 'img_trans_dict', 'ego_trans_dict',
                           'cam_intrinsic', 'cam2ego', 'pixel_wise_label'))
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        use_valid_flag=True,
        queue_length=0,
        num_frame_losses=1,
        seq_split_num=2,
        seq_mode=True),
    val=dict(
        type='NuSceneOcc',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/occ_infos_temporal_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='LoadOccGTFromFile', data_root='data/nuscenes/'),
            dict(
                type='CustomResizeCropFlipImage',
                data_aug_conf=dict(
                    resize_lim=(0.386, 0.55),
                    final_dim=(256, 704),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=False),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img'],
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'lidar2img', 'depth2img', 'cam2img',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'pcd_horizontal_flip', 'pcd_vertical_flip',
                                   'box_mode_3d', 'box_type_3d',
                                   'img_norm_cfg', 'pcd_trans', 'sample_idx',
                                   'pcd_scale_factor', 'pcd_rotation',
                                   'pts_filename', 'transformation_3d_flow',
                                   'scene_token', 'can_bus', 'ego2lidar',
                                   'prev_idx', 'next_idx', 'ego2global',
                                   'timestamp', 'img_trans_dict',
                                   'ego_trans_dict'))
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        queue_length=0,
        video_test_mode=True),
    test=dict(
        type='NuSceneOcc',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/occ_infos_temporal_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='LoadOccGTFromFile', data_root='data/nuscenes/'),
            dict(
                type='CustomResizeCropFlipImage',
                data_aug_conf=dict(
                    resize_lim=(0.386, 0.55),
                    final_dim=(256, 704),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=False),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img'],
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'lidar2img', 'depth2img', 'cam2img',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'pcd_horizontal_flip', 'pcd_vertical_flip',
                                   'box_mode_3d', 'box_type_3d',
                                   'img_norm_cfg', 'pcd_trans', 'sample_idx',
                                   'pcd_scale_factor', 'pcd_rotation',
                                   'pts_filename', 'transformation_3d_flow',
                                   'scene_token', 'can_bus', 'ego2lidar',
                                   'prev_idx', 'next_idx', 'ego2global',
                                   'timestamp', 'img_trans_dict',
                                   'ego_trans_dict'))
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        queue_length=0,
        video_test_mode=True),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=1265850,
    pipeline=[
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(type='LoadOccGTFromFile', data_root='data/nuscenes/'),
        dict(
            type='CustomResizeCropFlipImage',
            data_aug_conf=dict(
                resize_lim=(0.386, 0.55),
                final_dim=(256, 704),
                bot_pct_lim=(0.0, 0.0),
                rot_lim=(0.0, 0.0),
                H=900,
                W=1600,
                rand_flip=True),
            training=False),
        dict(
            type='NormalizeMultiviewImage',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='PadMultiViewImage', size_divisor=32),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(
                    type='Collect3D',
                    keys=['img'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'depth2img', 'cam2img',
                               'pad_shape', 'scale_factor', 'flip',
                               'pcd_horizontal_flip', 'pcd_vertical_flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'scene_token',
                               'can_bus', 'ego2lidar', 'prev_idx', 'next_idx',
                               'ego2global', 'timestamp', 'img_trans_dict',
                               'ego_trans_dict'))
            ])
    ])
checkpoint_config = dict(interval=14065, max_keep_ckpts=1)
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/only_pose_noise'
load_from = 'ckpts/r50_256x705_depth_pretrain.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
relative_relo_range = [-18.84, -18.84, -1.05, 18.84, 18.84, 1.05]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
point_class_names = [
    'ignore', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]
num_gpus = 2
batch_size = 1
num_iters_per_epoch = 14065
num_epochs = 90
bev_h_ = 100
bev_w_ = 100
num_points_in_pillar = 8
space_in_shape = [8, 100, 100]
space_out_shape = [16, 200, 200]
num_cams = 6
num_levels = 3
final_dim = (256, 704)
embed_dims = 72
num_heads = 9
num_frame_losses = 1
use_temporal = True
queue_length = 0
video_test_mode = True
num_memory = 4
voxel2bev = True
bev_dim = 126
time_range = [-2.3, 0.0]
model = dict(
    type='ViewFormer',
    use_grid_mask=True,
    video_test_mode=True,
    use_temporal=True,
    num_frame_backbone_grads=1,
    num_frame_head_grads=1,
    num_frame_losses=1,
    depth_supvise=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=3,
        add_extra_convs='on_output',
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='ViewFormerHead',
        pc_range=[-40, -40, -1.0, 40, 40, 5.4],
        num_levels=3,
        final_dim=(256, 704),
        in_channels=256,
        bev_h=100,
        bev_w=100,
        num_points_in_pillar=8,
        time_range=[-2.3, 0.0],
        use_mask_lidar=False,
        use_mask_camera=True,
        use_temporal=True,
        num_memory=4,
        bev_dim=126,
        relative_relo_range=[-18.84, -18.84, -1.05, 18.84, 18.84, 1.05],
        out_space3D_feat=False,
        space3D_net_cfg=dict(
            in_channels=72,
            bev_dim=126,
            feat_channels=32,
            in_shape=[8, 100, 100],
            out_shape=[16, 200, 200],
            num_classes=18),
        transformer=dict(
            type='ViewFormerTransformer',
            decoder=dict(
                type='ViewFormerTransformerDecoder',
                num_layers=4,
                return_intermediate=True,
                transformerlayers=dict(
                    type='ViewFormerTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='StreamTemporalAttn',
                            pc_range=[-40, -40, -1.0, 40, 40, 5.4],
                            num_levels=4,
                            embed_dims=126,
                            num_heads=9,
                            data_from_dict=True,
                            voxel2bev=True,
                            voxel_dim=72,
                            num_points=4),
                        dict(
                            type='ViewAttn',
                            pc_range=[-40, -40, -1.0, 40, 40, 5.4],
                            with_ffn=True,
                            num_levels=3,
                            embed_dims=72,
                            num_heads=9,
                            num_points=1)
                    ],
                    operation_order=('cross_attn', 'cross_attn')))),
        loss_prob=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=3.0),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_lovasz=dict(type='LovaszLoss', loss_weight=1.0)))
ida_aug_conf = dict(
    resize_lim=(0.386, 0.55),
    final_dim=(256, 704),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=True)
optimizer = dict(
    type='AdamW',
    lr=5e-05,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.25))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    min_lr_ratio=0.001)
runner = dict(type='IterBasedRunner', max_iters=1265850)
auto_resume = False
gpu_ids = range(0, 2)
