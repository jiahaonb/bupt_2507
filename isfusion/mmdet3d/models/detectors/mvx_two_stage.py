# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import warnings
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from .. import builder
from .base import Base3DDetector


@DETECTORS.register_module()
class MVXTwoStageDetector(Base3DDetector):
    """
    MVXTwoStageDetector (多模态体素网络两阶段检测器) 的基类。
    这是一个功能强大的检测器，能够同时处理点云和图像数据，
    并通过两阶段的方式进行3D目标检测。
    """

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        """
        初始化函数，根据配置文件构建模型的各个组件。
        
        点云处理分支 (pts):
        - pts_voxel_layer: 体素化层，将原始点云转换为规整的体素网格。
        - pts_voxel_encoder: 体素编码器，为每个非空体素提取特征。
        - pts_middle_encoder: 中间编码器，通常将稀疏的体素特征转换为密集的鸟瞰图(BEV)特征图。
        - pts_fusion_layer: 特征融合层，用于融合来自不同模态(如图像)的特征。
        - pts_backbone: 点云主干网络，在BEV特征图上进行深度特征提取。
        - pts_neck: 点云颈部网络 (如FPN)，融合多尺度特征。
        - pts_bbox_head: 3D检测头，根据最终特征预测3D边界框。

        图像处理分支 (img):
        - img_backbone: 图像主干网络 (如ResNet)，提取图像特征。
        - img_neck: 图像颈部网络 (如FPN)，融合多尺度图像特征。
        - img_rpn_head: 图像区域提议网络 (RPN)。
        - img_roi_head: 图像感兴趣区域 (RoI) 头。
        """
        super(MVXTwoStageDetector, self).__init__(init_cfg=init_cfg)

        # 构建点云处理分支的各个模块
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(
                pts_voxel_encoder)
        else:
            self.pts_voxel_encoder = None
        if pts_middle_encoder:
            self.pts_middle_encoder = builder.build_middle_encoder(
                pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        else:
            self.pts_backbone = None

        if pts_fusion_layer:
            self.pts_fusion_layer = builder.build_fusion_layer(
                pts_fusion_layer)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        if pts_bbox_head:
            # 将训练和测试配置传递给检测头
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)

        # 构建图像处理分支的各个模块
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if img_rpn_head is not None:
            self.img_rpn_head = builder.build_head(img_rpn_head)
        if img_roi_head is not None:
            self.img_roi_head = builder.build_head(img_roi_head)

        # fusion_encoder 可能是一个额外的融合模块
        self.fusion_encoder = None
        fusion_encoder = kwargs.get('fusion_encoder', None)
        if fusion_encoder is not None:
            self.fusion_encoder = builder.build_middle_encoder(fusion_encoder)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # 处理预训练权重的加载
        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
            pts_pretrained = pretrained.get('pts', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')

        if self.with_img_backbone:
            if img_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg')
                self.img_backbone.init_cfg = dict(
                    type='Pretrained', checkpoint=img_pretrained)
        if self.with_img_roi_head:
            if img_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg')
                self.img_roi_head.init_cfg = dict(
                    type='Pretrained', checkpoint=img_pretrained)

        if self.with_pts_backbone:
            if pts_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg')
                self.pts_backbone.init_cfg = dict(
                    type='Pretrained', checkpoint=pts_pretrained)


    @property
    def with_img_shared_head(self):
        """bool: 判断是否存在图像共享头。"""
        return hasattr(self,
                       'img_shared_head') and self.img_shared_head is not None

    @property
    def with_pts_bbox(self):
        """bool: 判断是否存在3D点云检测头。"""
        return hasattr(self,
                       'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        """bool: 判断是否存在2D图像检测头。"""
        return hasattr(self,
                       'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: 判断是否存在2D图像主干网络。"""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: 判断是否存在3D点云主干网络。"""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_fusion(self):
        """bool: 判断是否存在特征融合层。"""
        return hasattr(self,
                       'pts_fusion_layer') and self.fusion_layer is not None

    @property
    def with_img_neck(self):
        """bool: 判断是否存在图像颈部网络。"""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """bool: 判断是否存在点云颈部网络。"""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_img_rpn(self):
        """bool: 判断是否存在图像RPN。"""
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """bool: 判断是否存在图像RoI头。"""
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None

    @property
    def with_voxel_encoder(self):
        """bool: 判断是否存在体素编码器。"""
        return hasattr(self,
                       'voxel_encoder') and self.voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        """bool: 判断是否存在中间编码器。"""
        return hasattr(self,
                       'middle_encoder') and self.middle_encoder is not None

    def extract_img_feat(self, img, img_metas):
        """
        提取图像特征的完整流程。
        流程: img_backbone -> img_neck
        """
        if img is not None and img.dtype == torch.double:
            img = img.float()
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # 更新每个图像的真实输入尺寸
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            # 处理多视角图像输入 (B, N, C, H, W) -> (B*N, C, H, W)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            
            # 1. 通过主干网络提取特征
            img_feats = self.img_backbone(img)
        else:
            return None
        # 2. 通过颈部网络融合特征
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """
        提取点云特征的完整流程。
        流程: voxelize -> pts_voxel_encoder -> pts_middle_encoder -> pts_backbone -> pts_neck
        """
        if not self.with_pts_bbox:
            return None
        # 1. 体素化: 将点云从点空间转换到体素空间
        voxels, num_points, coors = self.voxelize(pts)
        # 2. 体素编码: 为每个体素生成特征向量。
        #    注意：这里可能会传入 img_feats，实现点云和图像特征的早期融合。
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                img_feats, img_metas)
        batch_size = coors[-1, 0] + 1
        # 3. 中间编码器: 将稀疏的体素特征转换为密集的BEV特征图
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        # 4. 点云主干网络: 在BEV特征图上进行深度特征提取
        x = self.pts_backbone(x)
        # 5. 点云颈部网络: 融合多尺度BEV特征
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_feat(self, points, img, img_metas):
        """
        从图像和点云中并行提取特征。
        这是模型特征提取的总入口。
        """
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """
        对点云进行动态体素化。
        
        Args:
            points (list[torch.Tensor]): 每个样本的点云列表。

        Returns:
            tuple[torch.Tensor]: 体素、每个体素的点数、体素坐标。
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            # 为每个样本的体素坐标添加 batch_id
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """
        训练时的前向传播函数。

        Args:
            points (list[torch.Tensor]): 输入的点云。
            img_metas (list[dict]): 样本的元信息。
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): 3D真值框。
            gt_labels_3d (list[torch.Tensor]): 3D真值标签。
            img (torch.Tensor): 输入的图像。
            ... (其他2D相关的真值和提议)

        Returns:
            dict: 包含所有分支损失的字典。
        """
        # 1. 并行提取图像和点云的特征
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        
        losses = dict()
        
        # 2. 处理点云分支，计算3D检测损失
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
            
        # 3. 处理图像分支，计算2D检测损失 (如果存在)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
            
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """
        点云分支的训练前向传播。
        流程: pts_feats -> pts_bbox_head -> loss
        """
        # 1. 将点云特征送入3D检测头得到预测输出
        outs = self.pts_bbox_head(pts_feats)
        # 2. 准备计算损失所需的输入 (预测+真值)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        # 3. 调用检测头的 loss 函数计算损失
        losses = self.pts_bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_img_train(self,
                          x,
                          img_metas,
                          gt_bboxes,
                          gt_labels,
                          gt_bboxes_ignore=None,
                          proposals=None,
                          **kwargs):
        """
        图像分支的训练前向传播 (类似于Faster R-CNN)。
        流程: img_feats -> RPN -> RoI Head -> loss
        """
        losses = dict()
        # 1. RPN 前向传播和损失计算
        if self.with_img_rpn:
            rpn_outs = self.img_rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas,
                                          self.train_cfg.img_rpn)
            rpn_losses = self.img_rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('img_rpn_proposal',
                                              self.test_cfg.img_rpn)
            proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.img_rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # 2. RoI Head (BBox Head) 前向传播和损失计算
        if self.with_img_bbox:
            img_roi_losses = self.img_roi_head.forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels,
                gt_bboxes_ignore, **kwargs)
            losses.update(img_roi_losses)

        return losses

    def simple_test_img(self, x, img_metas, proposals=None, rescale=False):
        """图像分支的无数据增强测试。"""
        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_metas,
                                                 self.test_cfg.img_rpn)
        else:
            proposal_list = proposals

        return self.img_roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def simple_test_rpn(self, x, img_metas, rpn_test_cfg):
        """RPN的测试函数。"""
        rpn_outs = self.img_rpn_head(x)
        proposal_inputs = rpn_outs + (img_metas, rpn_test_cfg)
        proposal_list = self.img_rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def simple_test_pts(self, x, img_metas, rescale=False):
        """点云分支的无数据增强测试。"""
        # 1. 将特征送入检测头获得预测
        outs = self.pts_bbox_head(x)
        if img_metas[0]is None:
            return outs
        # 2. 使用 get_bboxes 方法获取格式化的检测结果
        bbox_list = self.pts_bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # 3. 将结果转换为标准格式
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """
        无数据增强的完整测试函数。
        """
        # 1. 提取图像和点云特征
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        # 2. 如果存在点云分支，进行3D检测
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_metas, rescale=rescale)
            if img_metas[0]is None:
                return bbox_pts
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        # 3. 如果存在图像分支，进行2D检测
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """带数据增强的测试函数。"""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]

    def extract_feats(self, points, img_metas, imgs=None):
        """为多个样本（在数据增强测试中使用）提取特征。"""
        if imgs is None:
            imgs = [None] * len(img_metas)
        img_feats, pts_feats = multi_apply(self.extract_feat, points, imgs,
                                           img_metas)
        return img_feats, pts_feats

    def aug_test_pts(self, feats, img_metas, rescale=False, **kwargs):
        """带数据增强的点云分支测试函数。"""
        # 目前只支持单样本的增强测试
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            bbox_list = self.pts_bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # 合并多个增强结果
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.pts_bbox_head.test_cfg)
        return merged_bboxes

    def show_results(self, data, result, out_dir):
        """
        结果可视化。
        将点云和预测的3D边界框保存为文件，以便查看。
        """
        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id][
                    'pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                    'box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            file_name = osp.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'
            # 根据得分阈值筛选预测框
            inds = result[batch_id]['pts_bbox']['scores_3d'] > 0.1
            pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d'][inds]

            # 为了可视化，将点和框转换到深度相机坐标系
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                  == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(
                    f'Unsupported box_mode_3d {box_mode_3d} for convertion!')

            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            #调用可视化函数
            show_result(points, None, pred_bboxes, out_dir, file_name)
