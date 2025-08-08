import torch
from torch import nn

from detectors.utils.box_ops import box_cxcywh_to_xyxy


# TODO: need to go through and comment
class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api;

    Post processing class for DINO; only used during inference and evaluation
    """

    def __init__(self, num_select=100, nms_iou_threshold=-1) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    @torch.no_grad()
    def forward(
        self, outputs, target_sizes: torch.Tensor, not_to_xyxy=False, test=False
    ):
        """Perform the computation

        Args:
            outputs: raw outputs of the model
            target_sizes: tensor (b, 2) where 2 is the (h, w) of each images of the batch
                          for evaluation/inference:
                            this must be the original image size before any data augmentation
                            (including resizing and padding); this is because we want to map the
                            predicted bboxes back to the original image resolution of the input images;
                            this is important for metrics like mAP which are evaluated in terms of
                            original image_size; TODO: verify this
                          for visualization:
                            this should be the image size after data augmention/resizing but
                            before padding; this is because padding adds artificial borders and
                            we don't want to scale our predictions into padded areas that weren't
                            actually scene by the model(REMINDER: DETR first resizes the image such
                            that the shorter side to a fixed size (usually 800) and and the longer
                            side does not exceed 1333 (keeps aspect ratio), then the images are
                            converted to a NestedTensor and padded)
                           NOTE: not used during training
        """
        breakpoint()
        num_select = self.num_select
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), num_select, dim=1
        )
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:, :, 2:] = boxes[:, :, 2:] - boxes[:, :, :2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [
                nms(b, s, iou_threshold=self.nms_iou_threshold)
                for b, s in zip(boxes, scores)
            ]

            results = [
                {"scores": s[i], "labels": l[i], "boxes": b[i]}
                for s, l, b, i in zip(scores, labels, boxes, item_indices)
            ]
        else:
            results = [
                {"scores": s, "labels": l, "boxes": b}
                for s, l, b in zip(scores, labels, boxes)
            ]

        return results
