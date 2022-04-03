# This file is modified from https://github.com/tianweiy/CenterPoint

import torch
import torch.nn.functional as F
import numpy as np
import numba
from ...ops.iou3d_nms import iou3d_nms_utils
from scipy.optimize import linear_sum_assignment

def gaussian_radius(height, width, min_overlap=0.5):
    """
    Args:
        height: (N)
        width: (N)
        min_overlap:
    Returns:
    """
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / 2
    ret = torch.min(torch.min(r1, r2), r3)
    return ret


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian_to_heatmap(heatmap, center, radius, k=1, valid_mask=None):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom, radius - left:radius + right]
    ).to(heatmap.device).float()

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        if valid_mask is not None:
            cur_valid_mask = valid_mask[y - top:y + bottom, x - left:x + right]
            masked_gaussian = masked_gaussian * cur_valid_mask.float()

        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


@numba.jit(nopython=True)
def circle_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1
    return keep


def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    batch, num_class, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.flatten(2, 3), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_classes = (topk_ind // K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_classes, topk_ys, topk_xs


def decode_bbox_from_heatmap(heatmap, rot, center, center_z, dim,
                             point_cloud_range=None, voxel_size=None, feature_map_stride=None, vel=None,
                             circle_nms=False, score_thresh=None, post_center_limit_range=None, filter=False):
    if circle_nms:
        # TODO: not checked yet
        assert False, 'not checked yet'
        heatmap = _nms(heatmap)    

    # class label
    final_preds = heatmap.max(1, keepdims=False).indices
    final_scores = heatmap.max(1, keepdims=False).values

    # change size to real world metric
    center[:, 0, :] = center[:, 0, :] * feature_map_stride * voxel_size[0] + point_cloud_range[0]
    center[:, 1, :] = center[:, 1, :] * feature_map_stride * voxel_size[1] + point_cloud_range[1]
    # center[:, 2, :] = center[:, 2, :] * (self.post_center_range[5] - self.post_center_range[2]) + self.post_center_range[2]
    dim[:, 0, :] = dim[:, 0, :].exp()
    dim[:, 1, :] = dim[:, 1, :].exp()
    dim[:, 2, :] = dim[:, 2, :].exp()
    center_z = center_z - dim[:, 2:3, :] * 0.5  # gravity center to bottom center
    rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
    rot = torch.atan2(rots, rotc)
    
    if vel is None:
        final_box_preds = torch.cat([center, center_z, dim, rot], dim=1).permute(0, 2, 1)
    else:
        final_box_preds = torch.cat([center, center_z, dim, rot, vel], dim=1).permute(0, 2, 1)    
    
    predictions_dicts = []
    for i in range(heatmap.shape[0]):
        boxes3d = final_box_preds[i]
        scores = final_scores[i]
        labels = final_preds[i]
        predictions_dict = {
            'bboxes': boxes3d,
            'scores': scores,
            'labels': labels
        }
        predictions_dicts.append(predictions_dict)

    if filter is False:
        return predictions_dicts    
    
    # use score threshold
    if score_thresh is not None:
        thresh_mask = final_scores > score_thresh

    if post_center_limit_range is not None:
        post_center_limit_range = torch.tensor(
            post_center_limit_range, device=heatmap.device)
        mask = (final_box_preds[..., :3] >=
                post_center_limit_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <=
                post_center_limit_range[3:]).all(2)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            cmask = mask[i, :]
            if score_thresh:
                cmask &= thresh_mask[i]

            boxes3d = final_box_preds[i, cmask]
            scores = final_scores[i, cmask]
            labels = final_preds[i, cmask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

            predictions_dicts.append(predictions_dict)
    else:
        raise NotImplementedError(
            'Need to reorganize output as a batch, only '
            'support post_center_range is not None for now!')

    return predictions_dicts

def assign(bboxes, gt_bboxes, cls_pred, point_cloud_range):
        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bboxes.new_full((num_bboxes,),
                                           -1,
                                           dtype=torch.long)
        assigned_labels = bboxes.new_full((num_bboxes,),
                                          -1,
                                          dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            labels=assigned_labels
            return num_gts, assigned_gt_inds, None, labels

        # 2. compute the weighted costs
        # see mmdetection/mmdet/core/bbox/match_costs/match_cost.py
        gt_labels = (gt_bboxes[...,-1] - 1).type(torch.long)
        cls_cost = FocalLossCost(cls_pred[0].T, gt_labels)
        reg_cost = BBoxBEVL1Cost(bboxes, gt_bboxes, point_cloud_range, weight=0.25)
        iou = iou3d_nms_utils.boxes_iou3d_gpu(bboxes[...,:7], gt_bboxes[...,:7])
        iou_cost = IoU3DCost(iou, weight=0.25)

        # weighted sum of above three costs
        cost = cls_cost + reg_cost + iou_cost  #[NUM_PROPOSAL NUM_GT]

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)                   ##匈牙利算法
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bboxes.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bboxes.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        max_overlaps = torch.zeros_like(iou.max(1).values)
        max_overlaps[matched_row_inds] = iou[matched_row_inds, matched_col_inds]
        # max_overlaps = iou.max(1).values
        labels=assigned_labels
        return num_gts, assigned_gt_inds, max_overlaps, labels



def FocalLossCost(cls_pred, gt_labels, gamma=2, alpha=0.25, weight=0.15, eps=1e-12):
    """
    Args:
        cls_pred (Tensor): Predicted classfication logits
            in shape (num_query, d1, ..., dn), dtype=torch.float32.
        gt_labels (Tensor): Ground truth in shape (num_gt, d1, ..., dn),
            dtype=torch.long. Labels should be binary.
    Returns:
        Tensor: Focal cost matrix with weight in shape\
            (num_query, num_gt).
    """
    gt_labels = gt_labels.type(torch.long)
    cls_pred = cls_pred.sigmoid()
    neg_cost = -(1 - cls_pred + eps).log() * (
        1 - alpha) * cls_pred.pow(gamma)
    pos_cost = -(cls_pred + eps).log() * alpha * (
        1 - cls_pred).pow(gamma)

    cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
    return cls_cost * weight


def BBoxBEVL1Cost(bboxes, gt_bboxes, point_cloud_range, weight):
    pc_start = bboxes.new(point_cloud_range[0:2])
    pc_range = bboxes.new(point_cloud_range[3:5]) - bboxes.new(point_cloud_range[0:2])
    # normalize the box center to [0, 1]
    normalized_bboxes_xy = (bboxes[:, :2] - pc_start) / pc_range
    normalized_gt_bboxes_xy = (gt_bboxes[:, :2] - pc_start) / pc_range
    reg_cost = torch.cdist(normalized_bboxes_xy, normalized_gt_bboxes_xy, p=1)
    return reg_cost * weight


def IoU3DCost(iou, weight):
        iou_cost = - iou
        return iou_cost * weight


def encode(dst_boxes, code_size, point_cloud_range, feature_map_stride, voxel_size):
    targets = torch.zeros([dst_boxes.shape[0], code_size]).to(dst_boxes.device)
    targets[:, 0] = (dst_boxes[:, 0] - point_cloud_range[0]) / (feature_map_stride * voxel_size[0])
    targets[:, 1] = (dst_boxes[:, 1] - point_cloud_range[1]) / (feature_map_stride * voxel_size[1])
    # targets[:, 2] = (dst_boxes[:, 2] - self.post_center_range[2]) / (self.post_center_range[5] - self.post_center_range[2])
    targets[:, 3] = dst_boxes[:, 3].log()
    targets[:, 4] = dst_boxes[:, 4].log()
    targets[:, 5] = dst_boxes[:, 5].log()
    targets[:, 2] = dst_boxes[:, 2] + dst_boxes[:, 5] * 0.5  # bottom center to gravity center
    targets[:, 6] = torch.sin(dst_boxes[:, 6])
    targets[:, 7] = torch.cos(dst_boxes[:, 6])
    if code_size == 10:
        targets[:, 8:10] = dst_boxes[:, 7:9]
    return targets



'''    def assign(self, bboxes, gt_bboxes, gt_labels, cls_pred, train_cfg):
        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)   #gt个数， 预测的个数（200）

        # 1. assign -1 by default
        assigned_gt_inds = bboxes.new_full((num_bboxes,),
                                           -1,
                                           dtype=torch.long)
        assigned_labels = bboxes.new_full((num_bboxes,),
                                          -1,
                                          dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # see mmdetection/mmdet/core/bbox/match_costs/match_cost.py
        cls_cost = self.cls_cost(cls_pred[0].T, gt_labels)
        reg_cost = self.reg_cost(bboxes, gt_bboxes, train_cfg)
        iou = self.iou_calculator(bboxes, gt_bboxes)
        iou_cost = self.iou_cost(iou)

        # weighted sum of above three costs
        cost = cls_cost + reg_cost + iou_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bboxes.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bboxes.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        max_overlaps = torch.zeros_like(iou.max(1).values)
        max_overlaps[matched_row_inds] = iou[matched_row_inds, matched_col_inds]
        # max_overlaps = iou.max(1).values
        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)'''