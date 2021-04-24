import numpy as np
import torch
import torch.nn as nn

from mmpose.core.post_processing import flip_back
from mmpose.models.necks import GlobalAveragePooling
from ..registry import HEADS
from .heatmap_1d_head import Heatmap1DHead
from .heatmap_3d_head import Heatmap3DHead
from .multilabel_classification_head import MultilabelClassificationHead


@HEADS.register_module()
class Interhand3DHead(nn.Module):
    """Interhand 3D head of paper ref: Gyeongsik Moon. "InterHand2.6M: A
    Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single
    RGB Image".

    Args:
        keypoint_head_cfg (dict): Configs of Heatmap3DHead for hand
        keypoints estimation.
        root_head_cfg (dict): Configs of Heatmap1DHead for relative
        hand root depth estimation.
        hand_type_head_cfg (dict): Configs of MultilabelClassificationHead
        for hand type classification.
    """

    def __init__(self,
                 keypoint_head_cfg,
                 root_head_cfg,
                 hand_type_head_cfg,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        # build heads
        self.right_hand_head = Heatmap3DHead(**keypoint_head_cfg)
        self.left_hand_head = Heatmap3DHead(**keypoint_head_cfg)
        self.root_head = Heatmap1DHead(**root_head_cfg)
        self.hand_type_head = MultilabelClassificationHead(
            **hand_type_head_cfg)
        self.neck = GlobalAveragePooling()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self):
        self.left_hand_head.init_weights()
        self.right_hand_head.init_weights()
        self.root_head.init_weights()
        self.hand_type_head.init_weights()

    def get_loss(self, output, target, target_weight):
        """Calculate loss for hand keypoint heatmaps, relative root depth and
        hand type.

        Args:
            output (list[Tensor]): a list of outputs from multiple heads.
            target (list[Tensor]): a list of targets for multiple heads.
            target_weight (list[Tensor]): a list of targets weight for
            multiple heads.
        """
        losses = dict()
        losses['right_hand_loss'] = self.right_hand_head.get_loss(
            output[0], target[0], target_weight[0])['heatmap_loss']
        losses['left_hand_loss'] = self.left_hand_head.get_loss(
            output[1], target[1], target_weight[1])['heatmap_loss']
        losses['rel_root_loss'] = self.root_head.get_loss(
            output[2], target[2], target_weight[2])['value_loss']
        losses['hand_type_loss'] = self.hand_type_head.get_loss(
            output[3], target[3], target_weight[3])['classification_loss']
        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for hand keypoint heatmaps, relative root depth
        and hand type.

        Args:
            output (list[Tensor]): a list of outputs from multiple heads.
            target (list[Tensor]): a list of targets for multiple heads.
            target_weight (list[Tensor]): a list of targets weight for
            multiple heads.
        """

        acc = {}

        acc_right_hand = self.right_hand_head.get_accuracy(
            output[0], target[0], target_weight[0])
        for key in acc_right_hand:
            acc[f'right_{key}'] = acc_right_hand[key]

        acc_left_hand = self.left_hand_head.get_accuracy(
            output[1], target[1], target_weight[1])
        for key in acc_left_hand:
            acc[f'left_{key}'] = acc_left_hand[key]

        acc['acc_hand_type'] = self.hand_type_head.get_accuracy(
            output[3], target[3], target_weight[3])['acc_classification']
        return acc

    def forward(self, x):
        """Forward function."""
        outputs = []
        outputs.append(self.right_hand_head(x))
        outputs.append(self.left_hand_head(x))
        x = self.neck(x)
        outputs.append(self.root_head(x))
        outputs.append(self.hand_type_head(x))
        return outputs

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output (list[np.ndarray]): list of output hand keypoint
            heatmaps, relative root depth and hand type.

        Args:
            x (torch.Tensor[NxKxHxW]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """

        output = self.forward(x)

        if flip_pairs is not None:
            # concat heatmaps of right and left hand
            heatmap_3d = torch.cat([output[0], output[1]], dim=1)
            N, K, D, H, W = heatmap_3d.shape
            # reshape 3D heatmap to 2D heatmap
            heatmap_3d = heatmap_3d.reshape(N, K * D, H, W)
            # 2D heatmap flip
            heatmap_3d_flipped_back = flip_back(
                heatmap_3d.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.right_hand_head.target_type)
            # reshape back to 3D heatmap
            heatmap_3d_flipped_back = heatmap_3d_flipped_back.reshape(
                N, K, D, H, W)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                heatmap_3d_flipped_back[...,
                                        1:] = heatmap_3d_flipped_back[..., :-1]
            # split to right hand and left hand heatmaps
            output[0] = heatmap_3d_flipped_back[:, :21]
            output[1] = heatmap_3d_flipped_back[:, 21:]

            # flip relative hand root depth
            output[2] = -output[2].detach().cpu().numpy()

            # flip hand type
            hand_type = output[3].detach().cpu().numpy()
            hand_type_flipped_back = hand_type.copy()
            hand_type_flipped_back[:, 0] = hand_type[:, 1]
            hand_type_flipped_back[:, 1] = hand_type[:, 0]
            output[3] = hand_type_flipped_back
        else:
            output = [out.detach().cpu().numpy() for out in output]

        return output

    def decode(self, img_metas, output, **kwargs):
        """Decode hand keypoint, relative root depth and hand type.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
                - "heatmap3d_depth_bound": depth bound of hand keypoint
                 3D heatmap
                - "root_depth_bound": depth bound of relative root depth
                 1D heatmap


            output (list[np.ndarray]): model predicted 3D heatmaps, relative
            root depth and hand type.
        """

        batch_size = len(img_metas)
        heatmap3d_depth_bound = np.ones(batch_size, dtype=np.float32)
        root_depth_bound = np.ones(batch_size, dtype=np.float32)
        for i in range(batch_size):
            heatmap3d_depth_bound[i] = img_metas[i]['heatmap3d_depth_bound']
            root_depth_bound[i] = img_metas[i]['root_depth_bound']

        # decode 3D heatmaps of right and left hand keypoints respectively
        result = self.right_hand_head.decode(img_metas, output[0], **kwargs)
        result_left_hand = self.left_hand_head.decode(img_metas, output[0],
                                                      **kwargs)
        # concat keypoints of right and left hand
        keypoints_3d = np.concatenate(
            [result['preds'], result_left_hand['preds']], axis=1)
        # transform keypoint depth to camera space
        keypoints_3d[:,
                     3] = (keypoints_3d[:, 3] / self.right_hand_head.depth_size
                           - 0.5) * heatmap3d_depth_bound
        keypoints_3d = keypoints_3d[:, :, :3]
        result['preds'] = keypoints_3d

        # decode relative hand root depth
        result_root = self.root_head.decode(img_metas, output[2], **kwargs)
        # transform depth to camera space
        result['rel_root_depth'] = (
            result_root['values'] / self.root_head.heatmap_size -
            0.5) * root_depth_bound

        # decode hand type
        result_hand_type = self.hand_type_head.decode(img_metas, output[3],
                                                      **kwargs)
        result['hand_type'] = result_hand_type['labels']
        return result
