import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ..model_utils import centernet_trans_utils
from ...utils import loss_utils

import mayavi.mlab as mlab
import sys
from visual_utils.visualize_utils import draw_scenes

#transformer
from ..backbones_2d.det_seg_modules.transformer import TransformerDecoderLayer

class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, head_conv = 64, init_bias=-2.19, final_kernel = 1, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv1d(input_channels, head_conv, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=False),
                    nn.BatchNorm1d(head_conv),
                    nn.ReLU(inplace=True)
                ))
            fc_list.append(nn.Conv1d(head_conv, output_channels, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding



class CenterHeadTrans(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            )
        )


        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

        ###########################add ################################
        self.nms_kernel_size  = self.model_cfg.NMS_KERNEL_SIZE
        self.pos_weight = -1 
        self.auxiliary = True
        self.bias = 'auto'
        self.transformer_p = self.model_cfg.TRANSFORMER
        self.initialize_by_heatmap = self.model_cfg.INITIALIZE_BY_HEATMAP
        self.sampling = False
        self.num_proposals = self.model_cfg.NUM_PROPOSALS
        self.num_decoder_layers = self.model_cfg.NUM_DECODER_LAYER 

        if self.initialize_by_heatmap:
            self.learnable_query_pos = False
        else:
            self.learnable_query_pos = True

        if self.initialize_by_heatmap:
            layers = []
            layers.append(nn.Sequential(
            nn.Conv2d(
                self.model_cfg.SHARED_CONV_CHANNEL, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias= False
            ),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(inplace=True),
        ))
            layers.append(nn.Conv2d(
                self.model_cfg.SHARED_CONV_CHANNEL, self.num_class, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ))
            self.heatmap_head = nn.Sequential(*layers)
            self.class_encoding = nn.Conv1d(self.num_class, self.model_cfg.SHARED_CONV_CHANNEL, 1)
        else:
            # query feature
            self.query_feat = nn.Parameter(torch.randn(1, self.model_cfg.SHARED_CONV_CHANNEL, self.num_proposals))
            self.query_pos = nn.Parameter(torch.rand([1, self.num_proposals, 2]), requires_grad=self.learnable_query_pos)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    self.model_cfg.SHARED_CONV_CHANNEL, self.transformer_p.NHEAD, self.transformer_p.FFN_CHANNEL,
                    self.transformer_p.DROUPOUT, self.transformer_p.ACTIVATION,
                    self_posembed=PositionEmbeddingLearned(2, self.model_cfg.SHARED_CONV_CHANNEL),
                    cross_posembed=PositionEmbeddingLearned(2, self.model_cfg.SHARED_CONV_CHANNEL),
                ))

        # Prediction Head
        # self.prediction_heads = nn.ModuleList()
        # for i in range(self.num_decoder_layers):
        #     heads = copy.deepcopy(self.model_cfg.COMMONHEAD)
        #     heads.update(dict(heatmap=(self.num_classes, self.num_heatmap_convs)))
        #     self.prediction_heads.append(FFN(self.model_cfg.SHARED_CONV_CHANNEL, heads, bias=self.bias))


        self.prediction_heads = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for i in range(self.num_decoder_layers):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.prediction_heads.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
                )
            )


        self.init_weights()
        #self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.grid_size[0] // self.feature_map_stride
        y_size = self.grid_size[1] // self.feature_map_stride
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None


    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()
    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.transformer_p.BN_MOMENTUM


    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.WeightedSmoothL1Loss(
                code_weights=self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('code_weights', None)))
        self.add_module('cls_loss_func', loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0))

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, pred_dict, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        num_proposals = pred_dict['center'].shape[-1]            #proposal数量
        # get pred boxes, carefully ! donot change the network outputs
        batch_hm = copy.deepcopy(pred_dict['hm'].detach())
        batch_center = copy.deepcopy(pred_dict['center'].detach())
        batch_center_z = copy.deepcopy(pred_dict['center_z'].detach())
        batch_dim = copy.deepcopy(pred_dict['dim'].detach())
        batch_rot = copy.deepcopy(pred_dict['rot'].detach())
        if 'vel' in pred_dict.keys():
            batch_vel = copy.deepcopy(pred_dict['vel'].detach())
        else:
            batch_vel = None


        boxes_dict = centernet_trans_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot=batch_rot,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                circle_nms=(self.model_cfg.POST_PROCESSING.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=self.model_cfg.POST_PROCESSING.SCORE_THRESH,
                post_center_limit_range=self.model_cfg.POST_PROCESSING.POST_CENTER_LIMIT_RANGE
            )  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']      #[NUM_PROPOSAL, 9]
        gt_bboxes_tensor = gt_boxes.to(batch_hm.device)
        # each layer should do label assign seperately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = [] #for layer>1, but this only has 1 layer
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1), :]
            score_layer = batch_hm[..., self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1)]   #proposal的heatmap分数

            num_gts, assigned_gt_inds, max_overlaps, labels = centernet_trans_utils.assign(
                    bboxes = bboxes_tensor_layer, gt_bboxes = gt_bboxes_tensor, cls_pred = score_layer, point_cloud_range=self.point_cloud_range)
        
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = bboxes_tensor.new_zeros(bboxes_tensor.shape[0], dtype=torch.uint8)
        assert len(pos_inds) + len(neg_inds) == num_proposals
        
        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, 10]).to(batch_center.device)
        bbox_weights = torch.zeros([num_proposals, 10]).to(batch_center.device)
        ious = max_overlaps.to(batch_center.device)
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long) + self.num_class
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1
        if gt_bboxes_tensor.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes_tensor).view(-1, 4)
        else:
            if len(gt_bboxes_tensor.shape) < 2:
                gt_bboxes_tensor = gt_bboxes_tensor.view(-1, 4)

            pos_gt_bboxes = gt_bboxes_tensor[pos_assigned_gt_inds.long(), :]
        
        if len(pos_inds) > 0:
            pos_bbox_targets = centernet_trans_utils.encode(
                pos_gt_bboxes, code_size = 10, 
                point_cloud_range=self.point_cloud_range, 
                feature_map_stride=self.feature_map_stride,
                voxel_size=self.voxel_size)
            
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_bboxes_tensor is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = (gt_bboxes_tensor[...,-1] - 1).type(torch.long)[pos_assigned_gt_inds]  #(1-10) -> (0-9)
            if self.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.pos_weight
        
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0


        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0]).to(labels.device)

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(gt_boxes.shape[0]):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        
        return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None]

    def assign_targets(self, gt_boxes, feature_map_size=None, pred_dict = None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        list_of_pred_dict = []
        for batch_idx in range(len(gt_boxes)):       #from dict to batch_list 
            pred_dict_new = {}
            for key in pred_dict.keys():
                pred_dict_new[key] = pred_dict[key][batch_idx:batch_idx + 1]
            list_of_pred_dict.append(pred_dict_new)
        assert len(gt_boxes) == len(list_of_pred_dict)

        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'labels': [],
            'label_weights': [],
            'bbox_targets': [],
            'bbox_weights': [],
            'ious': [],
            'pos_inds': [],
            'mean_iou': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, labels_list, label_weights_list, bbox_targets_list = [], [], [], []
            bbox_weights_list, ious_list, pos_inds_list, mean_iou_list = [], [], [], []
            for bs_idx in range(batch_size):                                            #按照batch进行assign
                cur_gt_boxes = gt_boxes[bs_idx]                                         #当前batch的gt box
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])                       

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)    #去除了bg等不在cur_class_names的box
                
                #pred对应的label， pred label对应的权重（都是1）， pred对应的gt bbox（neg的都是0），pred对应的权重（neg的是0) 
                labels, label_weights, bbox_targets, bbox_weights, ious, pos_inds, mean_iou, heatmap = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                    pred_dict = list_of_pred_dict[bs_idx]
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                labels_list.append(labels.to(gt_boxes_single_head.device))
                label_weights_list.append(label_weights.to(gt_boxes_single_head.device))
                bbox_targets_list.append(bbox_targets.to(gt_boxes_single_head.device))
                bbox_weights_list.append(bbox_weights.to(gt_boxes_single_head.device))
                ious_list.append(ious.to(gt_boxes_single_head.device))
                pos_inds_list.append(pos_inds)
                mean_iou_list.append(mean_iou)


        ret_dict['heatmaps'] = torch.cat(heatmap_list, dim=0)
        ret_dict['labels'] = torch.cat(labels_list, dim=0)
        ret_dict['label_weights'] = torch.cat(label_weights_list, dim=0)
        ret_dict['bbox_targets'] = torch.cat(bbox_targets_list, dim=0)
        ret_dict['bbox_weights'] = torch.cat(bbox_weights_list, dim=0)
        ret_dict['ious'] = torch.cat(ious_list, dim=0)
        ret_dict['pos_inds'] = np.sum(pos_inds_list)
        ret_dict['mean_iou'] = np.mean(mean_iou_list)
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        labels = target_dicts['labels']
        label_weights = target_dicts['label_weights']
        bbox_weights = target_dicts['bbox_weights']
        bbox_targets = target_dicts['bbox_targets']
        tb_dict = {}
        loss = 0
        pred_dicts['heatmaps'] = self.sigmoid(pred_dicts['dense_heatmap'])
        hm_loss = self.hm_loss_func(pred_dicts['heatmaps'], target_dicts['heatmaps']) / (target_dicts['heatmaps'].eq(1).float().sum().item())
        hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        loss += hm_loss

        for idx_layer in range(self.num_decoder_layers if self.auxiliary else 1):
            if idx_layer == self.num_decoder_layers - 1 or (idx_layer == 0 and self.auxiliary is False):
                prefix = 'layer_-1'
            else:
                prefix = f'layer_{idx_layer}'

            #计算预测cls的loss
            layer_labels = labels[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].reshape(-1)
            target = F.one_hot(layer_labels, num_classes=self.num_class + 1)
            target = target[:, :self.num_class]
            layer_label_weights = label_weights[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].reshape(-1)
            layer_score = pred_dicts['hm'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_class)
            layer_loss_cls = self.cls_loss_func(layer_cls_score, target, weights = layer_label_weights).sum() / target_dicts['pos_inds']
            layer_loss_cls =layer_loss_cls * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            #计算回归bbox的loss
            layer_center = pred_dicts['center'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_height = pred_dicts['center_z'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_rot = pred_dicts['rot'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_dim = pred_dicts['dim'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            if 'vel' in pred_dicts.keys():
                layer_vel = pred_dicts['vel'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            #code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
            layer_bbox_weights = bbox_weights[:, idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, :]
            layer_reg_weights = layer_bbox_weights[...,0]
            layer_bbox_targets = bbox_targets[:, idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, :]
            layer_loss_bbox = self.reg_loss_func(preds[None, ...], layer_bbox_targets[None, ...], weights = layer_reg_weights[None, ...]).sum() / target_dicts['pos_inds']
            layer_loss_bbox *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']


            tb_dict[f'{prefix}_loss_cls'] = layer_loss_cls.item()
            tb_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox.item()
            loss += layer_loss_cls + layer_loss_bbox

        tb_dict['hm_loss'] = hm_loss.item()
        tb_dict['mean_iou'] = target_dicts['mean_iou']
        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict



    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        idx = 0
            
        batch_hm = pred_dicts['hm'].sigmoid()
        batch_center = pred_dicts['center']
        batch_center_z = pred_dicts['center_z']
        batch_dim = pred_dicts['dim'].exp()
        # batch_rot_cos = pred_dicts['rot'][:, 0].unsqueeze(dim=1)
        # batch_rot_sin = pred_dicts['rot'][:, 1].unsqueeze(dim=1)
        batch_rot = pred_dicts['rot']
        batch_vel = pred_dicts['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

        # final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
        #     heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
        #     center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
        #     point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
        #     feature_map_stride=self.feature_map_stride,
        #     K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
        #     circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
        #     score_thresh=post_process_cfg.SCORE_THRESH,
        #     post_center_limit_range=post_center_limit_range
        # )
        final_pred_dicts = centernet_trans_utils.decode_bbox_from_heatmap(
            heatmap=batch_hm, rot=batch_rot,
            center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
            point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
            feature_map_stride=self.feature_map_stride,
            circle_nms=(self.model_cfg.POST_PROCESSING.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
            score_thresh=self.model_cfg.POST_PROCESSING.SCORE_THRESH,
            post_center_limit_range=self.model_cfg.POST_PROCESSING.POST_CENTER_LIMIT_RANGE
        )

        for k, final_dict in enumerate(final_pred_dicts):
            final_dict['labels'] = self.class_id_mapping_each_head[idx][final_dict['labels'].long()]
            if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=final_dict['scores'], box_preds=final_dict['bboxes'],
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=None
                )

                final_dict['pred_boxes'] = final_dict['bboxes'][selected]
                final_dict['pred_scores'] = selected_scores
                final_dict['pred_labels'] = final_dict['labels'][selected]

            ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
            ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
            ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    
    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']                    #[BS, C', H, W] C'=512 H,W=180
        batch_size = spatial_features_2d.shape[0]                                 #BS
        lidar_feat = self.shared_conv(spatial_features_2d)                        #[BS, C, H, W] C=128 

        #################################
        # image to BEV
        #################################
        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)  # [BS, C, H*W]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)      # [BS, H*W, 2]
        
        if self.initialize_by_heatmap:
            dense_heatmap = self.heatmap_head(lidar_feat)                           #[BS, NUM_CLASS, H, W]
            heatmap = dense_heatmap.detach().sigmoid()
            padding = self.nms_kernel_size // 2
            local_max = torch.zeros_like(heatmap)
            # equals to nms radius = voxel_size * out_size_factor * kenel_size
            local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
            local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner        
            ## for Pedestrian & Traffic_cone in nuScenes
            if len(self.class_names)== 10:    #'nuScenes'
                local_max[:, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
                local_max[:, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
            else:  # for Pedestrian & Cyclist in Waymo
                local_max[:, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
                local_max[:, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
            heatmap = heatmap * (heatmap == local_max)
            heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)                  #[BS, NUM_CLASS, H*W]  heatmap.shape[-1]=H*W
            # top #num_proposals among all classes
            top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals] #proposals对应的idx序号 [BS, NUM_PROPOSAL]
            top_proposals_class = top_proposals // heatmap.shape[-1]                                                #proposals对应的class
            top_proposals_index = top_proposals % heatmap.shape[-1]                                                 #proposals对应的2d featmap位置
            #query_feat [BS, C, NUM_PROPOASL] C=128 NUM_PROPOASL=200
            query_feat = lidar_feat_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1) #得到对应proposal的feat
            self.query_labels = top_proposals_class                                                                 #proposals对应的class

            # add category embedding
            one_hot = F.one_hot(top_proposals_class, num_classes=self.num_class).permute(0, 2, 1)                   #[BS, NUM_CLASS, NUM_PROPOASL] class的onehot编码
            query_cat_encoding = self.class_encoding(one_hot.float())                                               #[BS, C, NUM_PROPOASL]
            query_feat += query_cat_encoding

            query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1) #得到proposal对应的bev position


        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################
        pred_dicts_list = []
        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](query_feat, lidar_feat_flatten, query_pos, bev_pos)       #trnasformer decoder  [BS, C, NUM_PROPOASL]

            # Prediction
            res_layer = self.prediction_heads[i](query_feat)                                      #head预测，总共有6个head:'center', 'center_z', 'dim', 'rot', 'vel','hm'
            res_layer['center'] = res_layer['center'] + query_pos.permute(0, 2, 1)                #最终center点位置，base+offset
            pred_dicts_list.append(res_layer)

            # for next level positional embedding
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)        


        pred_dicts = {}
        for key in pred_dicts_list[0].keys():                                                     #concat每个layer的输出
            pred_dicts[key] = torch.cat([pred_dict[key] for pred_dict in pred_dicts_list], dim=-1)
       
        pred_dicts['query_heatmap_score'] = heatmap.gather(index=top_proposals_index[:, None, :].expand(-1, self.num_class, -1), dim=-1)  # [bs, num_classes, num_proposals]
        pred_dicts['dense_heatmap'] = dense_heatmap


        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],             #gt, feature_map_size H,W
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None),              #stride=8
                pred_dict = pred_dicts                                                              #预测结果
            )
            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

                
        return data_dict
