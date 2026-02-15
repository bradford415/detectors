import torch
from torch import nn

from detectors.data.coco_utils import mscoco_label2category
from detectors.utils.box_ops import box_cxcywh_to_xyxy


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api;

    Post processing class for DINO; only used during inference and evaluation

    This postprocessing steps are as follows:
        1. choose topk `num_select` confidence `scores` across all queries and classes
        2. from the topk scores, extract the corresponding `labels` and `boxes`
        3. convert the `boxes` from relative [0, 1] cxcywh format to
          absolute [0, orig_img_w/h] xyxy (i.e., normalized coorindates to original image size)

    """

    def __init__(
        self,
        num_select: int = 100,
        contiguous_cat_ids: bool = False,
        nms_iou_threshold: int = -1,
    ) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold
        self.contiguous_cat_ids = contiguous_cat_ids

    @torch.no_grad()
    def forward(
        self, outputs: dict, target_sizes: torch.Tensor, not_to_xyxy=False, test=False
    ):
        """Perform the computation

        Args:
            outputs: raw output model predictions dict with contains the keys:
                        pred_logits: pred class logits for each class (b, num_queries, num_classes)
                        pred_boxes: pred bboxes normalized [0, 1] (b, num_queries, 4) where 4 = cxcywh
            target_sizes: tensor (b, 2) where 2 is the (h, w) of each images of the batch; used to scale
                          normalized coordinates to absolute coordinates i.e., [0, 1] -> [0, h/w]
                          for evaluation/inference:
                              this must be the original image dimensions before data augmentation
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

        Returns:
            a list of dictionaries for each image in the batch with the keys:
                scores: top `num_select` class probabilites which represent confidence (num_select,);
                        the top values are selected across num_queries*num_classes, therefore,
                        tehcnically a single query can have multiple classes predictions
                labels: the class_id predictions chosen by topk class probs (num_select,)
                boxes: the aboslute bboxes [0, orig_img_w/h] predictions chosen by topk class probs
                       (num_select, 4) where 4 = (x1, y1, x2, y2) in absolute coordinates
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # flatten the class logits for each query and pick the top num_select scores across
        # all classes and queries; it feels a bit weird to flatten the logits, rather than
        # taking the max for each query, however, from my understanding this approach allows for higher
        # recall, we don't want to miss a high-confidence class just because that query already had
        # a high score from something else
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), num_select, dim=1
        )
        scores = topk_values

        # decode the topk indices to which queries they belong to `topk_boxes` and which
        # class they belong to `labels`
        # NOTE: out_logits.shape[2] is the number of classes
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_cxcywh_to_xyxy(out_bbox)

        # was in rt detr code but pretty sure it's not used
        # if test:
        #     print("do not delete me")
        #     assert not not_to_xyxy
        #     boxes[:, :, 2:] = boxes[:, :, 2:] - boxes[:, :, :2]

        # NOTE: RT-Detr remaps coco categories to sequential in the dataloader but then
        #       here maps back to the original class ids; setting contiguous class ids is required
        #       or else you receive a cuda memory access error
        if self.contiguous_cat_ids:
            labels = (
                torch.tensor(
                    [mscoco_label2category[int(x.item())] for x in labels.flatten()]
                )
                .to(boxes.device)
                .reshape(labels.shape)
            )

        # extract the topk boxes coords in xyxy format
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # convert normalized bbox preds from [0, 1] to absolute [0, height/width] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # Not used, here just in case; DETRs say they don't need nms but sometimes still do
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
            #  package
            results = [
                {"scores": s, "labels": l, "boxes": b}
                for s, l, b in zip(scores, labels, boxes)
            ]

        return results
