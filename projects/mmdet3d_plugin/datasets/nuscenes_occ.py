import torch
import numpy as np
import os
from pandas import Timestamp
from tqdm import tqdm
from mmdet3d.datasets import NuScenesDataset
import mmcv
from mmdet.datasets import DATASETS
from nuscenes.eval.common.utils import Quaternion
from mmcv.parallel import DataContainer as DC
import random
from nuscenes.utils.geometry_utils import transform_matrix
from .occ_metrics import Metric_mIoU, Metric_FScore, Metric_AveError

import math

import pickle

import tempfile #싱글 GPU 학습 중 평가 에러 수정

@DATASETS.register_module()
class NuSceneOcc(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    PointClassMapping = {
        'animal': 'ignore',
        'human.pedestrian.personal_mobility': 'ignore',
        'human.pedestrian.stroller': 'ignore',
        'human.pedestrian.wheelchair': 'ignore',
        'movable_object.debris': 'ignore',
        'movable_object.pushable_pullable': 'ignore',
        'static_object.bicycle_rack': 'ignore',
        'vehicle.emergency.ambulance': 'ignore',
        'vehicle.emergency.police': 'ignore',
        'noise': 'ignore',
        'static.other': 'ignore',
        'vehicle.ego': 'ignore',
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck',
        'flat.driveable_surface': 'driveable_surface',
        'flat.other': 'other_flat',
        'flat.sidewalk': 'sidewalk',
        'flat.terrain': 'terrain',
        'static.manmade': 'manmade',
        'static.vegetation': 'vegetation'
    }

    POINT_CLASS_GENERAL = ('noise', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child',
                           'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility',
                           'human.pedestrian.police_officer', 'human.pedestrian.stroller',
                           'human.pedestrian.wheelchair', 'movable_object.barrier',
                           'movable_object.debris', 'movable_object.pushable_pullable',
                           'movable_object.trafficcone', 'static_object.bicycle_rack',
                           'vehicle.bicycle', 'vehicle.bus.bendy', 'vehicle.bus.rigid',
                           'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance',
                           'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer',
                           'vehicle.truck', 'flat.driveable_surface', 'flat.other', 'flat.sidewalk',
                           'flat.terrain', 'static.manmade', 'static.other', 'static.vegetation', 'vehicle.ego')

    POINT_CLASS_SEG = ('ignore', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                       'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                       'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                       'vegetation')

    THING_CLASSES = ('barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
    'pedestrian', 'traffic_cone', 'trailer', 'truck')

    POINT_LABEL_MAPPTING = []
    for name in POINT_CLASS_GENERAL:
        POINT_LABEL_MAPPTING.append(POINT_CLASS_SEG.index(PointClassMapping[name]))
    POINT_LABEL_MAPPTING = np.array(POINT_LABEL_MAPPTING, dtype=np.int32)

    def __init__(self,
                 queue_length=4,
                 seq_mode=False,
                 seq_split_num=1,
                 num_frame_losses=1,
                 video_test_mode=True,
                 eval_fscore=False,
                 eval_vel=False,
                 eval_bev_vel=False,
                 voxel_vel_path=None,
                 use_lidar_coord=False,
                 sparse_vel=False,
                 vel_dim=2,
                 debug_visualize_gt=False, # for debug visualization
                 data_type=None,
                 anno_file_path=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_visualize_gt = debug_visualize_gt # for debug visualization
        self.eval_fscore = eval_fscore
        self.eval_vel = eval_vel
        self.eval_bev_vel = eval_bev_vel
        self.queue_length = queue_length
        self.num_frame_losses = num_frame_losses
        self.video_test_mode = video_test_mode
        self.seq_mode = seq_mode
        self.use_lidar_coord = use_lidar_coord
        self.data_type = data_type
        self.anno_file_path = anno_file_path

        # when self.data_type is None, its the occ3D dataset accually
        #assert self.data_type in ['occ3D', 'OpenOcc', 'surroundOcc']

        self.voxel_vel_path = voxel_vel_path
        self.sparse_vel = sparse_vel
        self.vel_dim = vel_dim

        self.data_infos = self.load_annotations(self.ann_file)

        # refer to streampetr
        if self.seq_mode:
            self.num_frame_losses = 1
            self.queue_length = 0
            self.seq_split_num = seq_split_num
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]['sweeps']) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)


    def load_annotations(self, ann_file, dataset_ratio=None):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        if not isinstance(ann_file, str):
            ann_file = self.ann_file

        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]

        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        # we only use streaming video training for now
        if self.queue_length != 0:
            raise NotImplementedError

        queue = []
        index_list = list(range(index - self.queue_length, index))
        index_list.append(index)
        index_list.sort()

        # get target frame info and aug matrix, we aply the same aug matrix to other fames in window
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        target_example = self.pipeline(input_dict)
        queue.append(target_example)

        info = self.union2target(queue, index_list.index(index))

        return info

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def union2target(self, queue, target_frame_idx):

        target_info = queue[target_frame_idx]
        if not self.test_mode:
            imgs_list = []
            metas_map = []
            label_dict = dict(
                voxel_semantics=[], #각 voxel의 GT 클래스 로딩
                mask_lidar=[],
                mask_camera=[],
            )
            target_scene_token = target_info['img_metas'].data['scene_token']
            for i, each in enumerate(queue):
                if each['img_metas'].data['scene_token'] != target_scene_token:
                    # pad target frame info
                    frame_info = target_info
                else:
                    frame_info = each

                imgs_list.append(frame_info['img'].data)
                metas_map.append(frame_info['img_metas'].data)

                if i >= (len(queue) - self.num_frame_losses):
                    for key in label_dict.keys():
                        label_dict[key].append(torch.from_numpy(frame_info[key]))

            target_info['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
            target_info['img_metas'] = DC(metas_map, cpu_only=True)

            for key in label_dict.keys():
                target_info[key] = DC(torch.stack(label_dict[key]), cpu_only=False, stack=True)

        else:
            num_aug_data = len(target_info['img'])
            aug_imgs_list = []
            aug_metas_map = []
            for i in range(num_aug_data):
                imgs_list = []
                metas_map = []
                target_scene_token = target_info['img_metas'][i].data['scene_token']
                for j, each in enumerate(queue):
                    if each['img_metas'][i].data['scene_token'] != target_scene_token:
                        # pad target frame info
                        imgs_list.append(target_info['img'][i].data)
                        metas_map.append(target_info['img_metas'][i].data)
                    else:
                        imgs_list.append(each['img'][i].data)
                        metas_map.append(each['img_metas'][i].data)
                aug_imgs_list.append(DC(torch.stack(imgs_list), cpu_only=False, stack=True))
                aug_metas_map.append(DC(metas_map, cpu_only=True))
            target_info['img'] = aug_imgs_list
            target_info['img_metas'] = aug_metas_map

        return target_info

    def get_data_info(self, index): #GT
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]

        # standard protocal modified from SECOND.Pytorch
        occ_gt_path = info['occ_gt_path'] if 'occ_gt_path' in info else None
        lc_occ_gt_path = info['lc_occ_gt_path'] if 'lc_occ_gt_path' in info else None
        input_dict = dict(
            occ_gt_path=occ_gt_path,
            lc_occ_gt_path=lc_occ_gt_path,
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            scene_token=info['scene_token'],
            timestamp=info['timestamp'] / 1e6,
            render_frames=info['render_frames'] if 'render_frames' in info else None,
        )

        if self.data_type == 'surroundOcc':
            # we dont use surroundOcc pkl, trick to support surroundOcc annotation
            lidar_file_name = os.path.split(info['lidar_path'])[1]
            occ_gt_path = os.path.join(self.anno_file_path, lidar_file_name+'.npy')
            input_dict['occ_gt_path'] = occ_gt_path


        lidar2ego_rotation = info['lidar2ego_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        ego2lidar = transform_matrix(translation=lidar2ego_translation, rotation=Quaternion(lidar2ego_rotation),
                                     inverse=True)
        input_dict['ego2lidar'] = ego2lidar

        ego2global_rotation = info['ego2global_rotation']
        ego2global_translation = info['ego2global_translation']
        ego2global = transform_matrix(translation=ego2global_translation, rotation=Quaternion(ego2global_rotation),
                                     inverse=False)
        input_dict['ego2global'] = ego2global


        if self.use_lidar_coord:
            # trick to support lidar coordinate annotations, such as OpenOcc, surroundOcc
            # cause our code use the key 'ego2...' for transformation, we just change the value as 'lidar2...', but keep the key
            lidar2global = ego2global @ np.linalg.inv(ego2lidar)
            input_dict['ego2global'] = lidar2global
            input_dict['ego2lidar'] = np.eye(4)


        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            cam2ego_rts = []

            pixel_wise_label = []
            img_semantic = []

            for cam_type, cam_info in info['cams'].items():
                data_path = cam_info['data_path']
                image_paths.append(data_path)
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                                  'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

                tmp_lidar2cam_rt = lidar2cam_rt.T
                cam2ego_rt = np.linalg.inv(ego2lidar) @ np.linalg.inv(tmp_lidar2cam_rt)
                cam2ego_rts.append(cam2ego_rt)

                # only for auxiliary depth loss training
                try:
                    # we use image-wise depth supervision only when we compare with FB-Occ in occ3D dataset
                    _, file_name = os.path.split(cam_info['data_path'])
                    view_point_label = np.fromfile(os.path.join(
                            self.data_root, 'depth_pano_seg_gt', f'{file_name}.bin'),
                                                dtype=np.float32,
                                                count=-1).reshape(-1, 5)

                    cam_gt_depth = view_point_label[:, :3]
                    cam_gt_pano = view_point_label[:, 3:4].astype(np.int32)
                    cam_sem_mask = self.POINT_LABEL_MAPPTING[cam_gt_pano // 1000]

                    pixel_wise_label.append(np.concatenate([
                        cam_gt_depth,
                        cam_sem_mask.astype(np.float32)], axis=-1))

                except:
                    pass


                if not self.test_mode: # for seq_mode
                    prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
                else:
                    prev_exists = None

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    cam2ego=cam2ego_rts,
                    pixel_wise_label=pixel_wise_label,
                    prev_exists=prev_exists,
                ))


        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos



        # ===== GT 객체 마스크 생성 추가 =====
        if occ_gt_path is not None:
            occ_gt_full_path = os.path.join(self.data_root, occ_gt_path)
            occ_npz = np.load(occ_gt_full_path)
            
            # 정확한 키 사용 (데이터셋 구조에 따라 'semantics' 또는 'occupancy' 등 선택)
            if 'semantics' in occ_npz:
                occ_gt = occ_npz['semantics']
            elif 'occupancy' in occ_npz:
                occ_gt = occ_npz['occupancy']
            else:
                raise KeyError(f"Unexpected keys in {occ_gt_full_path}: {occ_npz.files}")
            input_dict['occupancy_gt'] = occ_gt

            object_mask_gt = ((occ_gt > 0) & (occ_gt <= 10)).astype(np.float32)
            # Z축 최대값을 취하여 BEV 객체 존재 여부만 남김 (H,W)
            bev_object_mask_gt = np.max(object_mask_gt, axis=-1)  # (H,W)
            # (1,H,W) 형태로 차원 추가
            bev_object_mask_gt = np.expand_dims(bev_object_mask_gt, axis=0)
            input_dict['object_mask_gt'] = bev_object_mask_gt.astype(np.float32)


            # # ----- GT 시각화 코드 (항상 실행) -----
            # # 저장 폴더를 절대 경로로 지정 (debug_gt_masks 폴더가 없으면 자동 생성)
            # save_dir = "/home/hyun/local_storage/code/object_vieeew/ViewFormer-Occ/work_dirs/Visualization/gt_masks_debug/"
            # os.makedirs(save_dir, exist_ok=True)
            # # info 딕셔너리에 sample_idx 또는 token이 있다면 사용, 없으면 'unknown'
            # token = input_dict.get('sample_idx', 'unknown')
            # import matplotlib.pyplot as plt
            # # BEV 마스크(채널 0)를 grayscale 이미지로 저장 (여러 채널일 경우 반복문 추가 가능)
            # plt.imsave(os.path.join(save_dir, f"gt_mask_{token}.png"), bev_object_mask_gt[0], cmap='gray')
            # # -----------------------------------


            # # ----- 원본 이미지 시각화 (항상 실행) -----
            # # 원본 이미지 경로는 input_dict['img_filename']에 저장되어 있다고 가정.
            # # 만약 여러 카메라 이미지가 있다면, 여기서는 첫 번째 이미지만 저장.
            # img_save_dir = "/home/hyun/local_storage/code/object_vieeew/ViewFormer-Occ/work_dirs/Visualization/gt-images/"
            # os.makedirs(img_save_dir, exist_ok=True)
            # if 'img_filename' in input_dict:
            #     # input_dict['img_filename']가 리스트인 경우 모든 카메라 이미지를 저장하도록 수정 가능
            #     if isinstance(input_dict['img_filename'], list):
            #         # 예시: 첫 번째 카메라 이미지 사용 (원한다면 for문으로 모두 저장 가능)
            #         img_path = input_dict['img_filename'][0]
            #     else:
            #         img_path = input_dict['img_filename']
            #     # 이미지 로드를 위해 OpenCV 사용 (이미지는 BGR이므로 RGB 변환)
            #     import cv2
            #     img = cv2.imread(img_path)
            #     if img is not None:
            #         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #         # 원본 이미지 저장 (GT mask와 동일한 token 사용)
            #         cv2.imwrite(os.path.join(img_save_dir, f"gt_img_{token}.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            #     else:
            #         print(f"Warning: 이미지 {img_path}를 로드할 수 없습니다.")
            # # ---------------------------------------


            # # ----- 6카메라 + BEV 마스크 콜라주 생성 (수정 후) -----
            # token = input_dict.get('sample_idx', 'unknown')
            # collage_dir = "/home/hyun/local_storage/code/object_vieeew/ViewFormer-Occ/work_dirs/Visualization/gt-camNmask/"
            # os.makedirs(collage_dir, exist_ok=True)

            # # 1) BEV 마스크 준비 (numpy, shape: (H, W))
            # bev_mask_img = bev_object_mask_gt[0]

            # # 2) 6개 카메라 이미지 로드 (input_dict['img_filename']가 리스트로 6개라 가정)
            # cam_paths = input_dict.get('img_filename', [])
            # cam_imgs = []
            # import mmcv
            # for p in cam_paths:
            #     if os.path.exists(p):
            #         cam_bgr = mmcv.imread(p)  # BGR
            #         cam_imgs.append(cam_bgr)
            #     else:
            #         cam_imgs.append(None)

            # # 3) 카메라를 원하는 순서대로 재배치하기
            # #    예: 상단 (2,0,1), 하단 (5,3,4) 순서
            # top_row_idx = [2, 0, 1]   # 상단 3개
            # bottom_row_idx = [5, 3, 4]  # 하단 3개

            # # 4) 서브플롯 콜라주 생성
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(3, 3, figsize=(16, 10))  # 사진 크기 확장
            # # 상단 타이틀 문구 변경
            # fig.suptitle(f"Multi-Camera + GT Mask (Token={token})", fontsize=16)

            # # (0,0) ~ (0,2): top_row_idx => [2,0,1]
            # for i in range(3):
            #     axs[0, i].axis('off')
            #     idx_cam = top_row_idx[i]
            #     if idx_cam < len(cam_imgs) and cam_imgs[idx_cam] is not None:
            #         axs[0, i].imshow(cam_imgs[idx_cam][..., ::-1])  # BGR->RGB
            #     else:
            #         axs[0, i].text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=12)

            # # 중단 행 (1,1)에 Object Mask GT
            # axs[1, 0].axis('off')
            # axs[1, 1].axis('off')
            # axs[1, 2].axis('off')

            # rotated_mask = bev_mask_img[::-1, ::-1]  # 2회전 -> 180도 회전

            # # 회전된 마스크로 시각화
            # axs[1, 1].imshow(rotated_mask, cmap='gray')
            # axs[1, 1].set_title("Object Mask GT", fontsize=14)

            # # (2,0) ~ (2,2): bottom_row_idx => [5,3,4]
            # for i in range(3):
            #     axs[2, i].axis('off')
            #     idx_cam = bottom_row_idx[i]
            #     if idx_cam < len(cam_imgs) and cam_imgs[idx_cam] is not None:
            #         axs[2, i].imshow(cam_imgs[idx_cam][..., ::-1])
            #     else:
            #         axs[2, i].text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=12)

            # out_path = os.path.join(collage_dir, f"{token}.png")
            # plt.savefig(out_path, bbox_inches='tight')
            # plt.close(fig)
            # # ---------------------------------------

        # ===== GT 객체 마스크 생성 완료 =====


        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            if self.video_test_mode:
                return self.prepare_test_data(idx)
            else:
                # for testing
                return self.prepare_train_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    # 멀티 GPU (원본 코드)
    def evaluate_miou(self, occ_results, runner=None, logger=None, show_dir=None, **eval_kwargs):
        if show_dir is not None:
            if not os.path.exists(show_dir):
                os.mkdir(show_dir)
            print('\nSaving output and gt in {} for visualization.'.format(show_dir))
            begin=eval_kwargs.get('begin',None)
            end=eval_kwargs.get('end',None)
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18 if not self.use_lidar_coord else 17, # please use data_type
            use_lidar_mask=False,
            use_image_mask=True,
            data_type=self.data_type)
        if self.eval_fscore:
            self.fscore_eval_metrics = Metric_FScore(
                leaf_size=10,
                threshold_acc=0.4,
                threshold_complete=0.4,
                voxel_size=[0.4, 0.4, 0.4],
                range=[-40, -40, -1, 40, 40, 5.4],
                void=[17, 255],
                use_lidar_mask=False,
                use_image_mask=True,
            )

        if self.eval_vel:
            self.vel_eval_metrics = Metric_AveError(
                value_range_list=[[-0.2, 0.2], [0.2, 1e3], [0.0, 1e3]],
                use_lidar_mask=False,
                use_image_mask=True,
            )

        if self.eval_bev_vel:
            self.bev_vel_eval_metrics = Metric_AveError(
                value_range_list=[[-0.2, 0.2], [0.2, 1e3], [0.0, 1e3]],
                use_lidar_mask=False,
                use_image_mask=True,
            )

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            if self.eval_vel or self.eval_bev_vel:
                occ_vel = occ_pred['occ_velocity'] if 'occ_velocity' in occ_pred else None
                bev_vel = occ_pred['bev_velocity'] if 'bev_velocity' in occ_pred else None
                occ_pred = occ_pred['occ_semantic']
            elif isinstance(occ_pred, dict):
                occ_pred = occ_pred['occ_semantic']

            info = self.data_infos[index]

            if self.data_type == 'surroundOcc':

                lidar_file_name = os.path.split(info['lidar_path'])[1]
                occ_gt_path = os.path.join(self.data_root, self.anno_file_path, lidar_file_name+'.npy')
                occ = np.load(occ_gt_path).astype(np.int64)
                occ_class = occ[:, -1]
                occ_class[occ_class == 0] = 255 # ignore, surroundOcc baseline ignore label 0

                W, H, Z = [200, 200, 16] # occ size
                occupancy_classes = 17
                gt_occupancy = np.ones(W*H*Z, dtype=np.uint8)*occupancy_classes
                occ_index = occ[:, 0] * H*Z + occ[:, 1] * Z + occ[:, 2] # (x, y, z) format
                gt_occupancy[occ_index] = occ_class

                gt_semantics = gt_occupancy.reshape(200, 200, 16)
                mask_camera = np.ones_like(gt_semantics, dtype=bool)
                mask_camera[gt_semantics == 255] = False # work around, use mask to achieve ignore
                mask_lidar = mask_camera

                # after ignore 0 in gt and pred, the class start from 1, we change it from 0
                # because mIoU calculate fn start from class 0
                gt_semantics = gt_semantics - 1
                occ_pred = occ_pred - 1

            elif self.data_type == 'OpenOcc' or 'lc_occ_gt_path' in info:
                # trick to support OpenOcc
                lc_occ_gt_path = info['lc_occ_gt_path']
                voxel_num = 200 * 200 * 16
                occupancy_classes = 16
                occ_gt_sparse = np.load(lc_occ_gt_path)
                occ_index = occ_gt_sparse[:, 0]
                occ_class = occ_gt_sparse[:, 1] 
                gt_occupancy = np.ones(voxel_num, dtype=np.uint8)*occupancy_classes
                gt_occupancy[occ_index] = occ_class

                occ_path, occ_name = os.path.split(info['lc_occ_gt_path'])
                invalid_path = os.path.join(occ_path, occ_name.split('.')[0] + '_invalid.npy')
                occ_invalid_index = np.load(invalid_path) # OccNet baseline use invalid_index in evaluation
                visible_mask = np.ones(voxel_num, dtype=bool)
                visible_mask[occ_invalid_index] = False

                gt_semantics = gt_occupancy.reshape(16, 200, 200).transpose(2, 1, 0)
                mask_camera = visible_mask.reshape(16, 200, 200).transpose(2, 1, 0)
                mask_lidar = mask_camera
            else:
                occ_gt = np.load(os.path.join(self.data_root, info['occ_gt_path']))
                if show_dir is not None:
                    if begin is not None and end is not None:
                        if index>= begin and index<end:
                            sample_token = info['token']
                            save_path = os.path.join(show_dir,str(index).zfill(4))
                            np.savez_compressed(save_path, pred=occ_pred, gt=occ_gt, sample_token=sample_token)
                    else:
                        sample_token=info['token']
                        save_path=os.path.join(show_dir,str(index).zfill(4))
                        np.savez_compressed(save_path,pred=occ_pred,gt=occ_gt,sample_token=sample_token)


                gt_semantics = occ_gt['semantics']
                mask_lidar = occ_gt['mask_lidar'].astype(bool)
                mask_camera = occ_gt['mask_camera'].astype(bool)
            occ_pred = occ_pred.squeeze(dim=0).cpu().numpy().astype(np.uint8)

            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)
            if self.eval_fscore:
                self.fscore_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)
            if self.eval_vel or self.eval_bev_vel:
                # work around
                assert self.voxel_vel_path is not None
                if self.sparse_vel:
                    W, H, Z = gt_semantics.shape
                    voxel_vel = np.ones((W*H*Z, 2)) * -1000
                    sparse_vel = np.fromfile(os.path.join(self.voxel_vel_path, info['token']+'.bin'), dtype=np.float16).reshape(-1, self.vel_dim)[:, :2]
                    sparse_idx = np.fromfile(os.path.join(self.voxel_vel_path, info['token']+'_idx.bin'), dtype=np.int32).reshape(-1)
                    voxel_vel[sparse_idx] = sparse_vel
                    gt_voxel_vel = voxel_vel.reshape(W, H, Z, 2)
                else:
                    voxel_vel_file_path = os.path.join(self.voxel_vel_path, info['token']+'.bin')
                    gt_voxel_vel = np.fromfile(voxel_vel_file_path, dtype=np.float16).reshape(*gt_semantics.shape, 2)
                gt_voxel_vel = gt_voxel_vel.astype(np.float32)
                valid_mask = gt_voxel_vel[..., 0] != -1000

                # flow is a TP error
                thing_mask = (occ_pred > 0) & (occ_pred <= 10) & (gt_semantics > 0) & (gt_semantics <= 10)
                valid_mask = valid_mask & thing_mask

                if self.eval_vel:
                    occ_vel = occ_vel.squeeze(dim=0).numpy()
                    occ_vel = occ_vel.astype(np.float32)
                    self.vel_eval_metrics.add_batch(occ_vel, gt_voxel_vel, mask_lidar & valid_mask, mask_camera & valid_mask)
                if self.eval_bev_vel:
                    bev_vel = bev_vel.squeeze(dim=0).numpy()
                    bev_vel = bev_vel.astype(np.float32)

                    gt_vel = torch.from_numpy(gt_voxel_vel)
                    gt_vel[~valid_mask] = 0.
                    vel_norm = gt_vel.norm(dim=-1)
                    _, max_idx = torch.max(vel_norm, dim=2)
                    gt_vel = gt_vel.gather(2, max_idx.unsqueeze(-1).unsqueeze(-1).repeat( 1, 1, 1, gt_vel.size(-1))).squeeze(2)
                    gt_vel = gt_vel.numpy()
                    self.bev_vel_eval_metrics.add_batch(bev_vel, gt_vel,
                                                        mask_lidar.any(axis=-1) & valid_mask.any(axis=-1),
                                                        mask_camera.any(axis=-1) & valid_mask.any(axis=-1))

        if logger is None and runner is not None:
            logger = runner.logger

        self.occ_eval_metrics.count_miou(logger=logger)
        if self.eval_fscore:
            self.fscore_eval_metrics.count_fscore(logger=logger)
        if self.eval_vel:
            self.vel_eval_metrics.count_ave_err(logger=logger, log_prifix='occ')
        if self.eval_bev_vel:
            self.bev_vel_eval_metrics.count_ave_err(logger=logger, log_prifix='bev')

    def format_results(self, occ_results,submission_prefix,**kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path=os.path.join(submission_prefix,'{}.npz'.format(sample_token))
            np.savez_compressed(save_path,occ_pred.squeeze(dim=0).numpy().astype(np.uint8))
        print('\nFinished.')





    # # 싱글 GPU 학습 중 평가 에러 수정
    # # --- NEW evaluate method added (FIXED RESULT COLLECTION) ---
    # # 이 함수가 이제 평가 후크에 의해 호출될 메트릭 계산의 핵심 함수.
    # # 기존 evaluate_miou 함수의 모든 핵심 로직이 이 안으로 들어옴.
    # def evaluate(self,
    #             results, # 모델 추론 결과 (predictions의 리스트)
    #             runner=None, # 후크에서 전달
    #             logger=None, # 후크에서 전달
    #             show_dir=None, # 후크/config에서 전달
    #             eval_config=None, # 후크/config에서 전달 (어떤 메트릭 계산할지 등)
    #              **eval_kwargs): # 기타 평가 관련 키워드 인자

    #     # 로거 설정 (기존 evaluate_miou에서 복사) <-- 맨 위로 이동.
    #     if logger is None and runner is not None:
    #         logger = runner.logger

    #     # 평가 메트릭 인스턴스 초기화 (기존 evaluate_miou에서 복사)
    #     # self.eval_fscore, self.eval_vel, self.eval_bev_vel 등의 플래그는 __init__에서 설정되어야 함.
    #     self.occ_eval_metrics = Metric_mIoU(
    #         num_classes=18 if not self.use_lidar_coord else 17,
    #         use_lidar_mask=False, use_image_mask=True, data_type=self.data_type)
    #     if self.eval_fscore:
    #         self.fscore_eval_metrics = Metric_FScore(
    #             leaf_size=10, threshold_acc=0.4, threshold_complete=0.4,
    #             voxel_size=[0.4, 0.4, 0.4], range=[-40, -40, -1, 40, 40, 5.4],
    #             void=[17, 255], use_lidar_mask=False, use_image_mask=True,
    #         )
    #     if self.eval_vel:
    #         self.vel_eval_metrics = Metric_AveError(
    #             value_range_list=[[-0.2, 0.2], [0.2, 1e3], [0.0, 1e3]],
    #             use_lidar_mask=False, use_image_mask=True,
    #         )
    #     if self.eval_bev_vel:
    #         self.bev_vel_eval_metrics = Metric_AveError(
    #             value_range_list=[[-0.2, 0.2], [0.2, 1e3], [0.0, 1e3]],
    #             use_lidar_mask=False, use_image_mask=True,
    #         )

    #     print('\nStarting Evaluation (Overridden method)...')

    #     # --- 모델 추론 결과 (results)를 순회하며 메트릭에 배치 추가하는 로직 ---
    #     # 이 부분은 기존 evaluate_miou의 메인 루프 내용을 가져옴.
    #     # evaluate_miou의 'occ_results' 변수 대신, 이 메서드의 'results' 인자를 사용.
    #     for index, occ_pred_raw in enumerate(tqdm(results)):
    #         info = self.data_infos[index] # GT 로딩을 위한 정보

    #         # --- GT 로딩 로직 (기존 evaluate_miou에서 완벽하게 복사) ---
    #         # data_type에 따른 GT 파일 로딩 및 마스크 처리 로직 전체를 복사.
    #         if self.data_type == 'surroundOcc':
    #             lidar_file_name = os.path.split(info['lidar_path'])[1]
    #             occ_gt_path = os.path.join(self.data_root, self.anno_file_path, lidar_file_name+'.npy')
    #             occ = np.load(occ_gt_path).astype(np.int64)
    #             occ_class = occ[:, -1]
    #             occ_class[occ_class == 0] = 255 # ignore, surroundOcc baseline ignore label 0

    #             W, H, Z = [200, 200, 16] # occ size
    #             occupancy_classes = 17
    #             gt_occupancy = np.ones(W*H*Z, dtype=np.uint8)*occupancy_classes
    #             occ_index = occ[:, 0] * H*Z + occ[:, 1] * Z + occ[:, 2] # (x, y, z) format
    #             gt_occupancy[occ_index] = occ_class

    #             gt_semantics = gt_occupancy.reshape(200, 200, 16)
    #             mask_camera = np.ones_like(gt_semantics, dtype=bool)
    #             mask_camera[gt_semantics == 255] = False # work around, use mask to achieve ignore
    #             mask_lidar = mask_camera

    #             # after ignore 0 in gt and pred, the class start from 1, we change it from 0
    #             # because mIoU calculate fn start from class 0
    #             gt_semantics_processed = gt_semantics - 1


    #         elif self.data_type == 'OpenOcc' or 'lc_occ_gt_path' in info:
    #             # trick to support OpenOcc
    #             lc_occ_gt_path = info['lc_occ_gt_path']
    #             voxel_num = 200 * 200 * 16
    #             occupancy_classes = 16
    #             occ_gt_sparse = np.load(lc_occ_gt_path)
    #             occ_index = occ_gt_sparse[:, 0]
    #             occ_class = occ_gt_sparse[:, 1]
    #             gt_occupancy = np.ones(voxel_num, dtype=np.uint8)*occupancy_classes
    #             gt_occupancy[occ_index] = occ_class

    #             occ_path, occ_name = os.path.split(info['lc_occ_gt_path'])
    #             invalid_path = os.path.join(occ_path, occ_name.split('.')[0] + '_invalid.npy')
    #             occ_invalid_index = np.load(invalid_path)
    #             visible_mask = np.ones(voxel_num, dtype=bool)
    #             visible_mask[occ_invalid_index] = False

    #             gt_semantics_processed = gt_occupancy.reshape(16, 200, 200).transpose(2, 1, 0)
    #             mask_camera = visible_mask.reshape(16, 200, 200).transpose(2, 1, 0)
    #             mask_lidar = mask_camera

    #         else: # Standard occ_gt_path loading (가장 흔한 경우)
    #             occ_gt = np.load(os.path.join(self.data_root, info['occ_gt_path']))
    #             gt_semantics_processed = occ_gt['semantics']
    #             mask_lidar = occ_gt['mask_lidar'].astype(bool)
    #             mask_camera = occ_gt['mask_camera'].astype(bool)


    #         # --- 모델 예측 결과 처리 및 메트릭에 배치 추가 로직 (기존 evaluate_miou에서 완벽하게 복사) ---
    #         # occ_pred_raw (모델 예측 결과 딕셔너리 또는 텐서)를 처리
    #         # occ_semantic, occ_velocity, bev_velocity 등을 추출하고 numpy/astype 변환
    #         # 각 메트릭 인스턴스(self.occ_eval_metrics 등)의 add_batch 메서드 호출

    #         occ_vel_raw = None
    #         bev_vel_raw = None
    #         occ_pred_semantic_raw = None

    #         # 예측 결과에서 semantic, velocity 등 추출
    #         if self.eval_vel or self.eval_bev_vel:
    #             occ_vel_raw = occ_pred_raw['occ_velocity'] if 'occ_velocity' in occ_pred_raw else None
    #             bev_vel_raw = occ_pred_raw['bev_velocity'] if 'bev_velocity' in occ_pred_raw else None
    #             occ_pred_semantic_raw = occ_pred_raw['occ_semantic']
    #         elif isinstance(occ_pred_raw, dict):
    #             occ_pred_semantic_raw = occ_pred_raw['occ_semantic']
    #         else:
    #             occ_pred_semantic_raw = occ_pred_raw # raw result is just semantic output

    #         # Semantic 예측 결과 numpy 변환 및 SurroundOcc/OpenOcc 클래스 오프셋 처리
    #         occ_pred_processed = occ_pred_semantic_raw.squeeze(dim=0).cpu().numpy().astype(np.uint8)
    #         if self.data_type == 'surroundOcc' or self.data_type == 'OpenOcc' or 'lc_occ_gt_path' in info:
    #             # NOTE: mIoU 계산을 위해 예측 결과에도 -1 오프셋을 적용해야 하는지 확인 필요
    #             occ_pred_processed = occ_pred_processed - 1


    #         # mIoU 메트릭에 배치 추가
    #         # 처리된 GT와 예측 결과를 사용.
    #         self.occ_eval_metrics.add_batch(occ_pred_processed, gt_semantics_processed, mask_lidar, mask_camera)

    #         # FScore 메트릭에 배치 추가 (eval_fscore 활성화 시)
    #         if self.eval_fscore:
    #             self.fscore_eval_metrics.add_batch(occ_pred_processed, gt_semantics_processed, mask_lidar, mask_camera)

    #         # Velocity 메트릭 처리 (eval_vel 또는 eval_bev_vel 활성화 시)
    #         if self.eval_vel or self.eval_bev_vel:
    #             assert self.voxel_vel_path is not None
    #             # --- Velocity GT 로딩/마스크 처리 (기존 evaluate_miou에서 완벽하게 복사) ---
    #             if self.sparse_vel:
    #                 W, H, Z = gt_semantics_processed.shape # 처리된 GT shape 사용
    #                 voxel_vel = np.ones((W*H*Z, 2)) * -1000
    #                 sparse_vel = np.fromfile(os.path.join(self.voxel_vel_path, info['token']+'.bin'), dtype=np.float16).reshape(-1, self.vel_dim)[:, :2]
    #                 sparse_idx = np.fromfile(os.path.join(self.voxel_vel_path, info['token']+'_idx.bin'), dtype=np.int32).reshape(-1)
    #                 voxel_vel[sparse_idx] = sparse_vel
    #                 gt_voxel_vel = voxel_vel.reshape(W, H, Z, 2)
    #             else: # Dense vel GT loading
    #                 voxel_vel_file_path = os.path.join(self.voxel_vel_path, info['token']+'.bin')
    #                 gt_voxel_vel = np.fromfile(voxel_vel_file_path, dtype=np.float16).reshape(*gt_semantics_processed.shape, 2).astype(np.float32) # 처리된 GT shape 사용


    #             valid_mask = gt_voxel_vel[..., 0] != -1000
    #             # 처리된 예측 결과를 사용.
    #             thing_mask = (occ_pred_processed > 0) & (occ_pred_processed <= 10) & (gt_semantics_processed > 0) & (gt_semantics_processed <= 10)
    #             valid_mask = valid_mask & thing_mask

    #             if self.eval_vel:
    #                 occ_vel_processed = occ_vel_raw.squeeze(dim=0).numpy().astype(np.float32)
    #                 self.vel_eval_metrics.add_batch(occ_vel_processed, gt_voxel_vel, mask_lidar & valid_mask, mask_camera & valid_mask)

    #             if self.eval_bev_vel:
    #                 bev_vel_processed = bev_vel_raw.squeeze(dim=0).numpy().astype(np.float32)
    #                 gt_vel_bev = torch.from_numpy(gt_voxel_vel)
    #                 gt_vel_bev[~valid_mask] = 0.
    #                 vel_norm = gt_vel_bev.norm(dim=-1)
    #                 _, max_idx = torch.max(vel_norm, dim=2)
    #                 gt_vel_bev = gt_vel_bev.gather(2, max_idx.unsqueeze(-1).unsqueeze(-1).repeat( 1, 1, 1, gt_vel_bev.size(-1))).squeeze(2).numpy()

    #                 # BEV 레벨 마스크 (기존 evaluate_miou에서 복사)
    #                 bev_valid_mask_lidar = mask_lidar.any(axis=-1) & valid_mask.any(axis=-1)
    #                 bev_valid_mask_camera = mask_camera.any(axis=-1) & valid_mask.any(axis=-1)

    #                 self.bev_vel_eval_metrics.add_batch(bev_vel_processed, gt_vel_bev,
    #                                                     bev_valid_mask_lidar,
    #                                                     bev_valid_mask_camera)


    #     # --- 메트릭 결과 계산 및 로깅 (count_* 메서드는 여기서 호출) ---
    #     # count_* 메서드는 결과를 로깅하고 메트릭 인스턴스 내부에 저장.
    #     self.occ_eval_metrics.count_miou(logger=logger)
    #     if self.eval_fscore:
    #         self.fscore_eval_metrics.count_fscore(logger=logger)
    #     if self.eval_vel:
    #         self.vel_eval_metrics.count_ave_err(logger=logger, log_prifix='occ')
    #     if self.eval_bev_vel:
    #         self.bev_vel_eval_metrics.count_ave_err(logger=logger, log_prifix='bev')


    #     # --- 메트릭 결과 취합 및 반환 (count_* 호출 후 인스턴스에서 결과 가져오기) ---
    #     # results_dict에 담아서 반환해야 평가 후크가 결과를 사용.
    #     results_dict = dict() # 결과 딕셔너리 초기화 (다시 초기화)

    #     # mIoU 결과 가져오기
    #     # Metric_mIoU 클래스는 내부 히스토그램을 .hist 속성에 저장하고
    #     # .per_class_iu(hist) 메서드로 per-class IoU를 계산한다고 가정.
    #     # 전체 mIoU는 이 per-class IoU 배열의 평균.
    #     # config의 metric='mIoU' 키에 맞게 'mIoU' 키에 값을 저장.
    #     if hasattr(self.occ_eval_metrics, 'hist') and hasattr(self.occ_eval_metrics, 'per_class_iu') and callable(self.occ_eval_metrics.per_class_iu):
    #         # 누적된 히스토그램을 가져와서 per_class_iu 메서드에 전달.
    #         histogram = self.occ_eval_metrics.hist # <-- 누적 히스토그램 속성 접근
    #         per_class_ious = self.occ_eval_metrics.per_class_iu(histogram) # <-- 히스토그램을 인자로 전달하여 메서드 호출

    #         # 메서드 호출 결과가 numpy 배열이나 리스트일 것으로 가정하고 평균 계산
    #         if isinstance(per_class_ious, (np.ndarray, list)):
    #             results_dict['mIoU'] = np.mean(per_class_ious)
    #         else:
    #             # 예상치 못한 타입이면 경고 및 기본값 설정
    #             print(f"Warning: unexpected type from per_class_iu(hist): {type(per_class_ious)}")
    #             results_dict['mIoU'] = 0.0 # 기본값 설정
    #     elif hasattr(self.occ_eval_metrics, 'miou'): # 만약 .miou 속성에 최종 mIoU가 저장된다면 (덜 흔하지만 확인)
    #         results_dict['mIoU'] = self.occ_eval_metrics.miou
    #     else:
    #         # mIoU 결과를 가져올 수 없는 경우
    #         print("Warning: Could not retrieve mIoU results from occ_eval_metrics instance.")
    #         results_dict['mIoU'] = 0.0 # 기본값 설정


    #     # FScore 결과 가져오기 (eval_fscore 활성화 시)
    #     # Metric_FScore 클래스가 결과를 .fscore 속성에 저장한다고 가정
    #     # 키 이름 'mFScore'는 프레임워크가 기대하는 이름과 일치해야함.
    #     if self.eval_fscore and hasattr(self.fscore_eval_metrics, 'fscore'):
    #         results_dict['mFScore'] = self.fscore_eval_metrics.fscore # mFScore 키 확인 필요


    #     # Velocity 결과 가져오기 (eval_vel 활성화 시)
    #     # Metric_AveError 클래스가 결과를 .average_error 속성에 저장한다고 가정
    #     if self.eval_vel and hasattr(self.vel_eval_metrics, 'average_error'):
    #         results_dict['occ_vel_error'] = self.vel_eval_metrics.average_error # 키 이름 확인 필요


    #     # BEV Velocity 결과 가져오기 (eval_bev_vel 활성화 시)
    #     # BEV velocity 메트릭 인스턴스가 별도의 평균 에러를 저장한다고 가정
    #     if self.eval_bev_vel and hasattr(self.bev_vel_eval_metrics, 'average_error'):
    #         if hasattr(self.bev_vel_eval_metrics, 'average_error'): # 안전을 위해 다시 확인
    #             results_dict['bev_vel_error'] = self.bev_vel_eval_metrics.average_error # 키 이름 확인 필요


    #     # 최종 결과 딕셔너리 반환 (이 줄은 그대로 유지)
    #     return results_dict


    # # --- MODIFIED format_results method ---
    # # 이 함수는 임시 디렉토리에 결과를 저장하고 그 객체를 반환.
    # def format_results(self, occ_results, submission_prefix=None, **kwargs): # submission_prefix 기본값=None 추가 (안전성)
    #     """평가 결과를 임시 디렉토리에 저장.""" # 주석 수정

    #     # 평가 계산에 필요한 중간 파일들을 저장할 임시 디렉토리를 항상 생성.
    #     # 상위 evaluate 메서드가 이 디렉토리를 정리할 것.
    #     # import tempfile # 파일 상단에 이미 추가.
    #     tmp_dir_obj = tempfile.TemporaryDirectory() # 임시 디렉토리 객체 생성
    #     eval_files_dir = tmp_dir_obj.name # 임시 디렉토리의 실제 경로

    #     # 임시 디렉토리가 존재하는지 확인
    #     mmcv.mkdir_or_exist(eval_files_dir)

    #     # 저장될 결과 파일 경로들을 저장할 리스트
    #     result_files = []

    #     for index, occ_pred in enumerate(tqdm(occ_results)):
    #         info = self.data_infos[index]
    #         sample_token = info['token']
    #         # 결과를 임시 디렉토리에 저장
    #         save_path = os.path.join(eval_files_dir, f'{sample_token}.npz') # f-string 사용 (가독성)
    #         np.savez_compressed(save_path, occ_pred.squeeze(dim=0).cpu().numpy().astype(np.uint8))
    #         result_files.append(save_path)

    #     print(f'\nFinished formatting {len(result_files)} results to temporary directory for evaluation.') # 더 자세한 출력

    #     # 저장된 파일 목록 (임시 디렉토리 내 경로)과 임시 디렉토리 객체 자체를 반환.
    #     # 상위 evaluate 메서드는 이 객체가 필요.
    #     return result_files, tmp_dir_obj # 파일 목록 리스트와 임시 디렉토리 객체 반환
    
