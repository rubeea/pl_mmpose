import numpy as np
import torch
from mmcv.parallel import collate, scatter

from mmpose.datasets.pipelines import Compose
from .inference import LoadImage, _box2cs, _xywh2xyxy, _xyxy2xywh


def _collate_pose_sequence(pose_results, with_track_id=True):
    """Reorganize multi-frame pose detection results into individual pose
    sequences.

    Notes:
        T: The temporal length of the pose detection results
        N: The number of the person instances
        K: The number of the keypoints
        C: The channel number of each keypoint

    Args:
        pose_results (List[List[Dict]]): Multi-frame pose detection results
            stored in a nested list. Each element of the outer list is the
            pose detection results of a single frame, and each element of the
            inner list is the pose information of one person, which contains:
                keypoints (ndarray[K, 2 or 3]): x, y, [score]
                track_id (int): unique id of each person, required when
                    ``with_track_id==True``
                bbox ((4, ) or (5, )): left, top, right, bottom, [score],
                    required when ``with_bbox==True``
        with_track_id (bool): If True, the element in pose_results is expected
            to contain "track_id", which will be used to gather the pose
            sequence of a person from multiple frames. Otherwise, the pose
            results in each frame are expected to have a consistent number and
            order of identities. Default is True.
    """
    T = len(pose_results)
    assert T > 0

    N = len(pose_results[-1])  # use identities in the last frame
    if N == 0:
        return []

    K, C = pose_results[-1][0]['keypoints'].shape

    track_ids = None
    if with_track_id:
        track_ids = [res['track_id'] for res in pose_results[-1]]

    pose_sequences = []
    for idx in range(N):
        pose_seq = dict(keypoints=np.zeros((T, K, C), dtype=np.float32))
        # gather static information
        for k, v in pose_results[-1][idx].items():
            if k != 'keypoints':
                pose_seq[k] = v
        pose_sequences.append(pose_seq)

    for t, frame in enumerate(pose_results):
        if with_track_id:
            id2idx = {res['track_id']: idx for idx, res in enumerate(frame)}
            indices = (id2idx.get(tid, None) for tid in track_ids)
        else:
            indices = range(N)

        for idx, idx_frame in enumerate(indices):
            if idx_frame is None:
                continue
            pose_sequences[idx]['keypoints'][t] = frame[idx_frame]['keypoints']

    return pose_sequences


def inference_pose_lifter_model(model,
                                pose_results_2d,
                                dataset,
                                with_track_id=True):
    """Inference 3D pose from 2D pose sequences using a pose lifter model.

    Args:
        model (nn.Module): The loaded pose lifter model
        pose_results_2d (List[List[dict]]): The 2D pose sequences stored in a
            nested list. Each element of the outer list is the 2D pose results
            of a single frame, and each element of the inner list is the 2D
            pose of one person, which contains:
                - "keypoints" (ndarray[K, 2 or 3]): x, y, [score]
                - "track_id" (int)
        dataset (str): Dataset name, e.g. 'Body3DH36MDataset'
        with_track_id: If True, the element in pose_results_2d is expected to
            contain "track_id", which will be used to gather the pose sequence
            of a person from multiple frames. Otherwise, the pose results in
            each frame are expected to have a consistent number and order of
            identities. Default is True.
    Returns:
        List[dict]: 3D pose inference results. Each element is the result of
            an instance, which contains:
            - "keypoints_3d" (ndarray[K,3]): predicted 3D keypoints
            - "keypoints" (ndarray[K, 2 or 3]): from the last frame in
                ``pose_results_2d``.
            - "track_id" (int): from the last frame in ``pose_results_2d``.
            If there is no valid instance, an empty list will be returned.
    """
    cfg = model.cfg
    test_pipeline = Compose(cfg.test_pipeline)

    flip_pairs = None
    if dataset == 'Body3DH36MDataset':
        flip_pairs = [[1, 4], [2, 5], [3, 6], [11, 14], [12, 15], [13, 16]]
    else:
        raise NotImplementedError()

    pose_sequences_2d = _collate_pose_sequence(pose_results_2d, with_track_id)

    if not pose_sequences_2d:
        return []

    batch_data = []
    for seq in pose_sequences_2d:
        pose_2d = seq['keypoints'].astype(np.float32)
        T, K, C = pose_2d.shape

        input_2d = pose_2d[..., :2]
        input_2d_visible = pose_2d[..., 2:3]
        if C > 2:
            input_2d_visible = pose_2d[..., 2:3]
        else:
            input_2d_visible = np.ones((T, K, 1), dtype=np.float32)

        # Dummy 3D input
        # This is for compatibility with configs in mmpose<=v0.14.0, where a
        # 3D input is required to generate denormalization parameters. This
        # part will be removed in the future.
        target = np.zeros((K, 3), dtype=np.float32)
        target_visible = np.ones((K, 1), dtype=np.float32)

        # Dummy image path
        # This is for compatibility with configs in mmpose<=v0.14.0, where
        # target_image_path is required. This part will be removed in the
        # future.
        target_image_path = None

        data = {
            'input_2d': input_2d,
            'input_2d_visible': input_2d_visible,
            'target': target,
            'target_visible': target_visible,
            'target_image_path': target_image_path,
            'ann_info': {
                'num_joints': K,
                'flip_pairs': flip_pairs
            }
        }

        data = test_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=len(batch_data))
    if next(model.parameters()).is_cuda:
        device = next(model.parameters()).device
        batch_data = scatter(batch_data, target_gpus=[device.index])[0]
    else:
        batch_data = scatter(batch_data, target_gpus=[-1])[0]

    with torch.no_grad():
        result = model(
            input=batch_data['input'],
            metas=batch_data['metas'],
            return_loss=False)

    poses_3d = result['preds']
    if poses_3d.shape[-1] != 4:
        assert poses_3d.shape[-1] == 3
        dummy_score = np.ones(
            poses_3d.shape[:-1] + (1, ), dtype=poses_3d.dtype)
        poses_3d = np.concatenate((poses_3d, dummy_score), axis=-1)
    pose_results = []
    for pose_2d, pose_3d in zip(pose_sequences_2d, poses_3d):
        pose_result = pose_2d.copy()
        pose_result['keypoints_3d'] = pose_3d
        pose_results.append(pose_result)

    return pose_results


def vis_3d_pose_result(model,
                       result,
                       img=None,
                       dataset='Body3DH36MDataset',
                       kpt_score_thr=0.3,
                       radius=8,
                       thickness=2,
                       show=False,
                       out_file=None):
    """Visualize the 3D pose estimation results.

    Args:
        model (nn.Module): The loaded model.
        result (list[dict])
    """
    if hasattr(model, 'module'):
        model = model.module

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    if dataset == 'Body3DH36MDataset':
        skeleton = [[1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8],
                    [8, 9], [9, 10], [10, 11], [9, 12], [12, 13], [13, 14],
                    [9, 15], [15, 16], [16, 17]]

        pose_kpt_color = palette[[
            9, 0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0
        ]]
        pose_limb_color = palette[[
            0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0
        ]]
    elif dataset == 'InterHand3DDataset':
        skeleton = [[1, 2], [2, 3], [3, 4], [4, 21], [5, 6], [6, 7], [7, 8],
                    [8, 21], [9, 10], [10, 11], [11, 12], [12, 21], [13, 14],
                    [14, 15], [15, 16], [16, 21], [17, 18], [18, 19], [19, 20],
                    [20, 21], [22, 23], [23, 24], [24, 25], [25, 42], [26, 27],
                    [27, 28], [28, 29], [29, 42], [30, 31], [31, 32], [32, 33],
                    [33, 42], [34, 35], [35, 36], [36, 37], [37, 42], [38, 39],
                    [39, 40], [40, 41], [41, 42]]

        pose_kpt_color = [[14, 128, 250], [14, 128, 250], [14, 128, 250],
                          [14, 128, 250], [80, 127, 255], [80, 127, 255],
                          [80, 127, 255], [80, 127, 255], [71, 99, 255],
                          [71, 99, 255], [71, 99, 255], [71, 99, 255],
                          [0, 36, 255], [0, 36, 255], [0, 36, 255],
                          [0, 36, 255], [0, 0, 230], [0, 0, 230], [0, 0, 230],
                          [0, 0, 230], [0, 0, 139], [237, 149, 100],
                          [237, 149, 100], [237, 149, 100], [237, 149, 100],
                          [230, 128, 77], [230, 128, 77], [230, 128, 77],
                          [230, 128, 77], [255, 144, 30], [255, 144, 30],
                          [255, 144, 30], [255, 144, 30], [153, 51, 0],
                          [153, 51, 0], [153, 51, 0], [153, 51, 0],
                          [255, 51, 13], [255, 51, 13], [255, 51, 13],
                          [255, 51, 13], [103, 37, 8]]

        pose_limb_color = [[14, 128, 250], [14, 128, 250], [14, 128, 250],
                           [14, 128, 250], [80, 127, 255], [80, 127, 255],
                           [80, 127, 255], [80, 127, 255], [71, 99, 255],
                           [71, 99, 255], [71, 99, 255], [71, 99, 255],
                           [0, 36, 255], [0, 36, 255], [0, 36, 255],
                           [0, 36, 255], [0, 0, 230], [0, 0, 230], [0, 0, 230],
                           [0, 0, 230], [237, 149, 100], [237, 149, 100],
                           [237, 149, 100], [237, 149, 100], [230, 128, 77],
                           [230, 128, 77], [230, 128, 77], [230, 128, 77],
                           [255, 144, 30], [255, 144, 30], [255, 144, 30],
                           [255, 144, 30], [153, 51, 0], [153, 51, 0],
                           [153, 51, 0], [153, 51, 0], [255, 51, 13],
                           [255, 51, 13], [255, 51, 13], [255, 51, 13]]

    else:
        raise NotImplementedError

    img = model.show_result(
        result,
        img,
        skeleton,
        radius=radius,
        thickness=thickness,
        pose_kpt_color=pose_kpt_color,
        pose_limb_color=pose_limb_color,
        show=show,
        out_file=out_file)

    return img


def inference_interhand_3d_model(model,
                                 img_or_path,
                                 det_results,
                                 bbox_thr=None,
                                 format='xywh',
                                 dataset='InterHand3DDataset'):
    """Inference a single image with a list of hand bounding boxes.

    num_bboxes: N
    num_keypoints: K

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str | np.ndarray): Image filename or loaded image.
        det_results (List[dict]): The 2D bbox sequences stored in a list.
            Each each element of the list is the bbox of one person, which
            contains:
                - "bbox" (ndarray[4 or 5]): The person bounding box,
                which contains 4 box coordinates (and score).
        dataset (str): Dataset name.
        format: bbox format ('xyxy' | 'xywh'). Default: 'xywh'.
            'xyxy' means (left, top, right, bottom),
            'xywh' means (left, top, width, height).

    Returns:
        List[dict]: 3D pose inference results. Each element is the result of
            an instance, which contains:
            - "keypoints_3d" (ndarray[K,3]): predicted 3D keypoints
            If there is no valid instance, an empty list will be returned.
    """

    assert format in ['xyxy', 'xywh']

    pose_results = []

    if len(det_results) == 0:
        return pose_results

    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in det_results])

    # Select bboxes by score threshold
    if bbox_thr is not None:
        assert bboxes.shape[1] == 5
        valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
        bboxes = bboxes[valid_idx]
        det_results = [det_results[i] for i in valid_idx]

    if format == 'xyxy':
        bboxes_xyxy = bboxes
        bboxes_xywh = _xyxy2xywh(bboxes)
    else:
        # format is already 'xywh'
        bboxes_xywh = bboxes
        bboxes_xyxy = _xywh2xyxy(bboxes)

    # if bbox_thr remove all bounding box
    if len(bboxes_xywh) == 0:
        return []

    cfg = model.cfg
    device = next(model.parameters()).device

    # build the data pipeline
    channel_order = cfg.test_pipeline[0].get('channel_order', 'rgb')
    test_pipeline = [LoadImage(channel_order=channel_order)
                     ] + cfg.test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    assert len(bboxes[0]) in [4, 5]

    if dataset == 'InterHand3DDataset':
        flip_pairs = [[i, 21 + i] for i in range(21)]
    else:
        raise NotImplementedError()

    batch_data = []
    for bbox in bboxes:
        center, scale = _box2cs(cfg, bbox)

        # prepare data
        data = {
            'img_or_path':
            img_or_path,
            'center':
            center,
            'scale':
            scale,
            'bbox_score':
            bbox[4] if len(bbox) == 5 else 1,
            'bbox_id':
            0,  # need to be assigned if batch_size > 1
            'dataset':
            dataset,
            'joints_3d':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'joints_3d_visible':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'rotation':
            0,
            'ann_info': {
                'image_size': np.array(cfg.data_cfg['image_size']),
                'num_joints': cfg.data_cfg['num_joints'],
                'flip_pairs': flip_pairs,
                'heatmap3d_depth_bound': cfg.data_cfg['heatmap3d_depth_bound'],
                'heatmap_size_root': cfg.data_cfg['heatmap_size_root'],
                'root_depth_bound': cfg.data_cfg['root_depth_bound']
            }
        }

        data = test_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # scatter not work so just move image to cuda device
        batch_data['img'] = batch_data['img'].to(device)
    # get all img_metas of each bounding box
    batch_data['img_metas'] = [
        img_metas[0] for img_metas in batch_data['img_metas'].data
    ]

    # forward the model
    with torch.no_grad():
        result = model(
            img=batch_data['img'],
            img_metas=batch_data['img_metas'],
            return_loss=False)

    poses_3d = result['preds']
    rel_root_depth = result['rel_root_depth']
    hand_type = result['hand_type']
    if poses_3d.shape[-1] != 4:
        assert poses_3d.shape[-1] == 3
        dummy_score = np.ones(
            poses_3d.shape[:-1] + (1, ), dtype=poses_3d.dtype)
        poses_3d = np.concatenate((poses_3d, dummy_score), axis=-1)

    # add relative root depth to left hand joints
    poses_3d[:, 21:, 2] += rel_root_depth

    # set joint scores according to hand type
    poses_3d[:, :21, 3] *= hand_type[:, [0]]
    poses_3d[:, 21:, 3] *= hand_type[:, [1]]

    pose_results = []
    for pose_3d, person_res, bbox_xyxy in zip(poses_3d, det_results,
                                              bboxes_xyxy):
        pose_res = person_res.copy()
        pose_res['keypoints_3d'] = pose_3d
        pose_res['bbox'] = bbox_xyxy
        pose_results.append(pose_res)

    return pose_results
