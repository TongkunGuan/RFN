# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import RBoxList
from maskrcnn_benchmark.structures.rboxlist_ops import boxlist_nms, cluster_nms
from maskrcnn_benchmark.structures.rboxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.rbox_coder import RBoxCoder
import numpy as np

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self, score_thresh=0.02, nms=0.5, detections_per_img=100, box_coder=None, nms_type="remove", shrink_margin=1.4
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
            nms_type: "remove" or "merge"
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = RBoxCoder(weights=(10., 10., 5., 5., 1.))
        self.box_coder = box_coder
        self.shrink_margin = shrink_margin
        self.nms_fn = boxlist_nms if nms_type == "remove" else cluster_nms
        self.update_cls = False
        self.update_box = False

    def forward(self, x, boxes, num_of_fwd_left=0):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """

        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        # proposals = concat_boxes
        # print("proposals 1:", concat_boxes.shape, class_prob.shape, box_regression.shape)
        # if not self.update_box:
        # print("rbox concat_boxes:", concat_boxes[:10])
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        # print("rbox proposals:", proposals[:10])
        # self.update_box = True
        # print("proposals 2:", proposals.shape, class_prob.shape, box_regression.shape)
        # else:
        #     proposals = torch.cat([concat_boxes] * 2, -1)

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape in zip(
            class_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            #boxlist = self.filter_results(boxlist, num_classes, num_of_fwd_left)
            results.append(boxlist)
            # print("boxlist:", boxlist.bbox.shape)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 5:(j + 1) * 5]`.
        """
        boxes = boxes.reshape(-1, 5)
        scores = scores.reshape(-1)
        boxlist = RBoxList(boxes, image_shape, mode="xywha")
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes, num_of_fwd_left):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 5)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]

            # print("scores_j:", np.unique(scores_j.data.cpu().numpy())[-10:])

            boxes_j = boxes[inds, j * 5 : (j + 1) * 5]
            boxlist_for_class = RBoxList(boxes_j, boxlist.size, mode="xywha")
            boxlist_for_class.add_field("scores", scores_j)

            if num_of_fwd_left == 0:
                boxlist_for_class.rescale(1. / self.shrink_margin)
                boxlist_for_class = self.nms_fn(
                    boxlist_for_class, self.nms, score_field="scores"
                )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.RBBOX_REG_WEIGHTS
    box_coder = RBoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    nms_type = cfg.MODEL.ROI_HEADS.NMS_TYPE
    shrink_margin = cfg.MODEL.RRPN.GT_BOX_MARGIN
    postprocessor = PostProcessor(
        score_thresh, nms_thresh, detections_per_img, box_coder, nms_type, shrink_margin
    )
    return postprocessor
