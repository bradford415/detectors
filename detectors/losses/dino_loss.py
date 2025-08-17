import copy

import torch
from torch import nn
from torch.nn import functional as F

from detectors.data.data import NestedTensor
from detectors.metrics import topk_accuracy
from detectors.models.components.matcher import build_matcher
from detectors.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from detectors.utils.distributed import get_world_size, is_dist_avail_and_initialized


# TODO: Go through and understand/comment
class SetCriterion(nn.Module):
    """This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]

        Args:
            outputs:
            targets:
            indices: list of indices (for each img in batch) containing tuples which represent
                     (pred_indices, target_indices); pred_indices are the location of the respective
                     query prediction and target_indices are the index of the gt target label;
                     e.g., target_indices = [0,1,2,3] and the image contains object_labels = [32, 4, 8, 9],
                           then target_indices will later index object_labels
            num_boxes: average number of gt boxes across all nodes; if using dn_queries this is multiplied
                       by the number of dn_groups since this creates this many replicas of the gt boxes

        Returns:
            a dictionary with the key:
                "loss_ce": which is the focal loss of queries/predictions
        """
        assert "pred_logits" in outputs

        # logits predictions (b, num_queries, max_class_id+1)
        src_logits = outputs["pred_logits"]

        # combine the batch_indices and query_indices for the targets in each image
        # `idx` is 2 element tuple (batch_inds, query_inds) with each element
        # shape: real_queries (sum(num_gt_targets_i), )
        #        dn_queries: (sum(num_dn_group*num_objects_i),) where i is the img_index of the batch
        idx = self._get_src_permutation_idx(indices)

        # extract the ground truth label for each query prediction across all images in the batch
        # shape: real_queries: (sum(num_gt_targets_i), )
        #        dn_queries: (sum(num_dn_group*num_objects_i),) where i is the img_index of the batch
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        # create a tensor of "no_object" classes for each query (b, num_queries) where
        # each element = num_classes (max_class_id+1)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        # overwrite with the correct gt class labels in the indices of the query predictions;
        # this gives us a full tensor gt class, where unmatched queries are background (b, num_queries);
        # note that `indices` input to the function does not necessarily have to be all the
        # queries, for example one call just passes the pos dn_queries;
        # NOTE: for pos_dn_queries, I think that since the 2nd half of the dn_group represents the
        #       negative dn_queries, the background class is good
        target_classes[idx] = target_classes_o

        # create a onehot tensor of the constructed gt targets;
        # NOTE: that we +1 to num_classes so when we form onehots, label num_classes can index
        #       the num_classes onehot tensor (or else we would get an out of bounds error I think);
        #       after, we slice this end off [..., :-1] because the focal loss does not care about
        #       the 'no-object/background' class
        # NOTE: based on this the onehot index [0] is just a dummy label since coco class IDs start at 1;
        #       this post mentions it: https://github.com/facebookresearch/detr/issues/108#issuecomment-674854977
        #       and a few replies down it explains that the network shouldn't suffer with a few dummy labels
        #       with no examples in the train set
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(
            dim=2, index=target_classes.unsqueeze(-1), value=1
        )

        # the only labels that have a background class are the negative dn_queries
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        # computes the focal loss (a scalar) for the current batch, averaged by the average
        # number of boxes across all processes and the number of dn_groups
        loss_ce = (
            sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * src_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            # NOTE: I think this is just used for logging, not in the total loss/gradient updates
            #       since this is not summed in the total loss
            losses["class_error"] = (
                100 - topk_accuracy(src_logits[idx], target_classes_o)[0]
            )
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients

        A perfect loss of 0.0 would be when the number of non `no-object` predicted classes is the number
        of ground-truth objects
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device

        # number of objects in each image in the batch (b,)
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )

        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)

        # compute the mean absolute error
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.

        Returns:
            a bbox loss dictionary with the keys:
                "loss_bbox": the l1 norm of the pred bboxes, averaged per bbox
                "loss_giou": the giou of the matched boxes, averged per bbox
                "loss_xy": the l1 norm of the cx, cy preds, averaged per bbox;
                           does not contribute to gradient updates
                "loss_xy": the l1 norm of the h, w preds, averaged per bbox;
                           does not contribute to gradient updates

        """
        assert "pred_boxes" in outputs

        # combine the batch_indices and query_indices for the targets in each image
        # `idx` is 2 element tuple (batch_inds, query_inds) with each element
        # (sum(num_dn_group*num_objects_i),) where i is the img_index of the batch
        idx = self._get_src_permutation_idx(indices)

        # extract the predicted boxes for each image in the batch specified by `idx`
        # (sum(num_dn_groups*num_objects_i),) where i represents the image idx in the batch
        src_boxes = outputs["pred_boxes"][idx]

        # extract the ground truth boxes for every prediction (sum(num_dn_groups*num_objects_i), 4)
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        # calculate the element wise absolute error (MAE without the mean) -> |pred_n - target_n|
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        assert src_boxes.shape == target_boxes.shape == loss_bbox.shape

        # average the bbox loss by the average number of boxes across all nodes times the
        # num_dn_groups for a per_box_loss
        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # compute the giou between the predicted and the target boxes; generalized_box_iou()
        # gives us a pairwise matrix (i.e., a single predicted box is compared to every gt box)
        # but during this loss computation only one predicted box can match to one gt box so taking
        # the diagonal gives us this 1-to-1 matching (i.e., pred[i] to gt[i], not pred[i] to pred[j]);
        # moreso, since giou ranges from -1 (bad) to 1 (good) we subtract 1 to convert it to a loss
        # i.e., lower giou means higher loss
        loss_giou = 1 - torch.diag(  # (sum(num_dn_groups*num_objects_i),)
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
        )

        # average the giou loss by the average number of boxes across all nodes times the
        # num_dn_groups for a per_box_loss
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss; wrapping this no_grad() does not add these operations
        # to computation graph which saves memory and allows for faster computation; technically
        # even without no_grad these would still not contribute to gradient updates because
        # when summing the total loss we do not include it, but this no_grad is nice to have for
        # the reasons above
        with torch.no_grad():
            losses["loss_xy"] = loss_bbox[..., :2].sum() / num_boxes
            losses["loss_hw"] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = NestedTensor.from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        """Permute predictions according to `indices`"""

        # store a tensor of the batch/sample index (sum(num_dn_group*num_objects_i),)
        # where i represents the img index in the batch
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )

        # store a tensor of the index of the query index locations for every image in the batch
        # (sum(num_dn_group*num_objects_i),)
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """This performs the loss computation.
        Args:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                     The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        Return:
            a dictionary of loss components; not all of these are used to propagate gradients;
            losses for the real, learnable queries (topk) from the last decoder layer output:
                `loss_ce`: predicted labels focal loss, normalized by the average num_gt_boxes per process
                `loss_bbox`: predicted bboxes l1 distance loss, normalized by the average num_gt_boxes per process
                `loss_giou`: predicted bboxes giou loss, normalized by the average num_gt_boxes per process
                `cardinality_error`: l1 distance (mae) of # of non `no-object` preds = # of gt objects;
                                     this is not counted in the total loss & does not propagate gradients
                `class_error`: the top 1 error for the predicted classes; not counted in total loss
                               and does not propagate gradients
            losses for the real, learnable queries (topk) from every decoder layer output except the last
            where i = range(num_decoder_layers-1):
                `loss_ce_i`: predicted labels focal loss, normalized by the average num_gt_boxes per process
                `loss_bbox_i`: predicted bboxes l1 distance loss, normalized by the average num_gt_boxes per process
                `loss_giou_i`: predicted bboxes giou loss, normalized by the average num_gt_boxes per process
                `cardinality_error_i`: l1 distance (mae) of # of non `no-object` preds = # of gt objects;
                                     this is not counted in the total loss & does not propagate gradients
                `loss_ce_dn`: predicted labels focal loss, normalized by the average num_gt_boxes per process
                `loss_bbox_dn`: predicted bboxes l1 distance loss, normalized by the average num_gt_boxes per process
                `loss_giou_dn`: predicted bboxes giou loss, normalized by the average num_gt_boxes per process
                `cardinality_error_dn`: l1 distance (mae) of # of non `no-object` preds = # of gt objects;
                                     this is not counted in the total loss & does not propagate gradients
            losses for the dn queries from the last decoder layer output:
            where i = range(num_decoder_layers-1):
                `loss_ce_dn_i`: predicted labels focal loss, normalized by the average num_gt_boxes per process
                `loss_bbox_dn_i`: predicted bboxes l1 distance loss, normalized by the average num_gt_boxes per process
                `loss_giou_dn_i`: predicted bboxes giou loss, normalized by the average num_gt_boxes per process
                `cardinality_error_dn_i`: l1 distance (mae) of # of non `no-object` preds = # of gt objects;
                                     this is not counted in the total loss & does not propagate gradients
        """
        # extract the non-auxiliary outputs; aux outputs are the intermediate decoder outputs
        # passed through the detection and class heads (every decoder output but the last)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        device = next(iter(outputs.values())).device

        # TODO: comment what this does
        indices = self.matcher(outputs_without_aux, targets)

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes;
        # this is useful for cases where an image in a batch has a lot of objects and another image
        # in the same batch has very few objects, this lets us calculate the loss per box, on average;
        # number of gt-boxes are summed across all proccesses and this sum is broadcasted to each
        # process (i.e., each process will have this same summed value)
        # NOTE: The number of predicted boxes can vary, so using ground truth count as normalization
        #       is more stable and meaningful
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)

        # divide by the number of processes to get an average number of boxes per node
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs["dn_meta"]

        if self.training and dn_meta and "output_known_lbs_bboxes" in dn_meta:
            # extract the dn_queries:
            #   `output_known_lbs_bboxes`: predicted bboxes & labels (last dec layer) & aux_predictions,
            #   and the `single_pad` and num_dn_groups (`scalar`) to index the pos/neg queries properly
            # NOTE: this single_pad is calculated from the max objects in the batch at the start of the
            #       train loop so that there's enough space for a pos & neg query for each gt object;
            #       images in the batch with less objects won't use this full range
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta)

            # store the indices of the pos/neg queries for each image in the batch
            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):

                # if the image has gt objects (not a negative image)
                if len(targets[i]["labels"]) > 0:

                    # create a tensor from [0, num_objects-1] * scalar (num_objects*scalar,)
                    t = torch.arange(0, len(targets[i]["labels"])).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()

                    # create tensor of offset indices for each dn_group, add the object indices, and flatten
                    # (scalar,) -> (scalar, 1) (scalar, num_objects), (scalar*num_objects,)
                    output_idx = (
                        torch.tensor(range(scalar)) * single_pad
                    ).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    # for negative images use an empty tensor
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                # store the indices of the pos/neg queries; the first half of the group is reserved for
                # pos queries and the 2nd half is reserved for neg queries
                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes = dn_meta["output_known_lbs_bboxes"]
            l_dict = {}

            # compute each of the desired losses for the positive and negative dn_queries; positive
            # dn_queries are lightly noised and are expected to recover the gt label while negative
            # dn_queries are heavly noised and expected to predict 'no_object' (Section 3..3); we
            # pass the positive dn_query_indices which use the 1st half of the dn_group and the
            # negative dn_queries use the 2nd half; a gt_label tensor is filled with the no_object
            # class_id and then the gt labels are filled in this tensor at dn_pos_idx
            # (default losses: "labels", "boxes", "cardinality")
            for loss in self.losses:
                kwargs = {}
                if "labels" in loss:
                    kwargs = {"log": False}
                l_dict.update(
                    self.get_loss(
                        loss,
                        output_known_lbs_bboxes,
                        targets,
                        dn_pos_idx,
                        # multiply by num_dn_groups so we can normalize per box; dn_groups is copy of
                        # the gt_boxes so if we don't account for this the loss could be overweighted
                        num_boxes * scalar,
                        **kwargs,
                    )
                )

            l_dict = {k + f"_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            # if denoising is not used, set all denoising losses to 0.0
            l_dict = dict()
            l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["loss_ce_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["loss_xy_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["loss_hw_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["cardinality_error_dn"] = torch.as_tensor(0.0).to("cuda")
            losses.update(l_dict)

        # compute the losses for the real, learnable queries (topk) for the last decoder layer output:
        #   1. `loss_ce`: predicted labels focal loss, averaged by num_gt_boxes across all proccesses
        #   2. `loss_bbox`: predicted bboxes l1 distance loss, averaged by num_gt_boxes across all proccesses
        #   3. `loss_giou`: predicted bboxes giou loss, averaged by num_gt_boxes across all proccesses
        #   4. `cardinality_error`: l1 distance (mae) of # of non `no-object` preds = # of gt objects;
        #                           this is not counted in the total loss & does not propagate gradients
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer
        if "aux_outputs" in outputs:

            # loop through each intermediate decoder predictions (num_dec_layers-1)
            for idx, aux_outputs in enumerate(outputs["aux_outputs"]):

                # match the auxiliary predictions with the gt targets
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)

                # compute losses for the real, learnable queries (topk) for the intermediate dec outputs:
                #   i = range(0, num_dec_layers-1)
                #   1. `loss_ce_i`: predicted labels focal loss
                #   2. `loss_bbox_i`: predicted bboxes l1 distance loss
                #   3. `loss_giou_i`: predicted bboxes giou loss
                #   4. `cardinality_error`: l1 distance (mae) of
                #                           # of non `no-object` preds = # of gt objects;
                #                           this is not counted in the total loss & does not
                #                           propagate gradients
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too csotly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

                # compute losses for the pos/neg, dn queries for the intermediate dec outputs:
                #   i = range(0, num_dec_layers-1)
                #   1. `loss_ce_dn_i`: predicted labels focal loss
                #   2. `loss_bbox_dn_i`: predicted bboxes l1 distance loss
                #   3. `loss_giou_dn_i`: predicted bboxes giou loss
                #   4. `cardinality_error`: l1 distance (mae) of
                #                           # of non `no-object` preds = # of gt objects;
                #                           this is not counted in the total loss & does not
                #                           propagate gradients
                if self.training and dn_meta and "output_known_lbs_bboxes" in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes["aux_outputs"][idx]
                    l_dict = {}
                    for loss in self.losses:
                        kwargs = {}
                        if "labels" in loss:
                            kwargs = {"log": False}

                        l_dict.update(
                            self.get_loss(
                                loss,
                                aux_outputs_known,
                                targets,
                                dn_pos_idx,
                                num_boxes * scalar,
                                **kwargs,
                            )
                        )

                    l_dict = {k + f"_dn_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    # if denoising is not used, set all denoising losses to 0.0
                    l_dict = dict()
                    l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["loss_ce_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["loss_xy_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["loss_hw_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["cardinality_error_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict = {k + f"_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # compute losses for the learnable queries (topk) encoder_output:
        #   1. `loss_ce_interm`: predicted labels focal loss
        #   2. `loss_bbox_interm`: predicted initial reference point l1 distance loss
        #   3. `loss_giou_interm`: predicted initial reference giou loss
        #   4. `cardinality_error`: l1 distance (mae) of
        #                           # of non `no-object` preds = # of gt objects;
        #                           this is not counted in the total loss & does not
        #                           propagate gradients
        if "interm_outputs" in outputs:
            interm_outputs = outputs["interm_outputs"]
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == "masks":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs = {"log": False}
                l_dict = self.get_loss(
                    loss, interm_outputs, targets, indices, num_boxes, **kwargs
                )
                l_dict = {k + f"_interm": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # skipped; enc output loss
        if "enc_outputs" in outputs:
            for i, enc_outputs in enumerate(outputs["enc_outputs"]):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, enc_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self, dn_meta):
        """Extract the dn_queries components such as predicted bboxes & labels, pad_size and
        # of denoising groups; these allow us to find the pos/neg denoising query indices
        """
        output_known_lbs_bboxes = dn_meta["output_known_lbs_bboxes"]
        num_dn_groups, pad_size = dn_meta["num_dn_group"], dn_meta["pad_size"]
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups


def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2
):
    """Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.



    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions (class logits) for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_boxes: total number of gt boxes for all images across all proccesses
                   multiplied by number of dn_groups; lets us normalize the loss
                   to get an average loss per box
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()

    # binary_cross_entropy take into account every class indepenently; even if the classes
    # that are not the correct one contribute to the loss; we do not use reduction so we can
    # handle each class independently (b, num_predictions, num_classes)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # this essentially says "how correct was the prediction?"
    p_t = prob * targets + (1 - prob) * (1 - targets)

    # apply the focal scaling to the standard bce -> (BCE * (1-pt) ** gamma) per class
    # main idea:
    #   if p_t is close to 1 -> model is confident -> the scaling factor (1-p_t)**gamma is low
    #   if p_t is close to 0- > model is wrong -> the loss is amplified
    loss = ce_loss * ((1 - p_t) ** gamma)  # (b, num_predictions, num_classes)

    # balance the correct and incorrect classes such that incorrect classes have higher weighting;
    # correct classes get weighted (0.25) and incorrect classes get weighted (1 - 0.25 = 0.75)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # average the loss over every prediction per class (b, num_classes), sum across every element
    # to get a scalar and then finally normalize by
    return loss.mean(1).sum() / num_boxes


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    NOTE: only used in loss_masks

    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if __torchvision_need_compat_flag < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(
            input, size, scale_factor, mode, align_corners
        )


def create_dino_loss(
    num_classes: int,
    num_decoder_layers: int,
    aux_loss: bool,
    two_stage_type: str,
    loss_args: dict[str, any],
    matcher_args: dict[str, any],
    device: torch.device,
):
    """Builds the dino loss function along with the Matcher

    Args:
        num_classes: number of classes in the dataset ontology
        num_decoder_layers: number of decoder layers in the TransformerDecoder
        aux_loss: whether to use auxiliary losses for each decoder layer
        two_stage_type:
        loss_args:
        matcher_args: parameters for the hungarian matcher
        device: device used to compute the loss on

    Returns:
        the loss function used in DINO
    """

    # build the hungarian matcher object to match predicted object detections to ground-truth
    # annotations in a one-to-one manner; since DETR predicts a fixed-size set of predictions,
    # we need a way to assign each ground-truth object to one unique prediction to compute a loss;
    # the matcher ensures each predicted object is assigned to at most one ground-truth object;
    # the hungarian algorithm finds the optimal (lowest-cost) matching between the predicted boxes
    # and gt boxes; the cost function for matching is weight combination of the classification cost,
    # bbox L1 distance, and generalized iou cost;
    # NOTE: DINO uses the focal loss for the classification cost (unlike original DETR) to help
    #       handle class imbalance more effectively
    matcher = build_matcher(**matcher_args)

    # prepare weight dict; create a copy to save these params w/o the denoising params
    weight_dict = {
        "loss_ce": loss_args["cls_loss_coef"],
        "loss_bbox": loss_args["bbox_loss_coef"],
        "loss_giou": loss_args["giou_loss_coef"],
    }
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    # for denoising training, add the same params as above but for the dn key
    weight_dict["loss_ce_dn"] = loss_args["cls_loss_coef"]
    weight_dict["loss_bbox_dn"] = loss_args["bbox_loss_coef"]
    weight_dict["loss_giou_dn"] = loss_args["giou_loss_coef"]

    # NOTE: removed mask keys bc unused

    # copy dict w/ denoising params
    clean_weight_dict = copy.deepcopy(weight_dict)

    # update the `weight_dict` w/ a copy of the dictionary keys for each decoder
    # layer (except the last one) with the the decoder_layer_index appended
    # (e.g., "loss_ce_0, loss_ce_1, ..., loss_ce_<num_decoder_layers-1>")
    if aux_loss:

        num_decoder_layers = num_decoder_layers

        aux_weight_dict = {}
        for i in range(num_decoder_layers - 1):
            aux_weight_dict.update(
                {k + f"_{i}": v for k, v in clean_weight_dict.items()}
            )
        weight_dict.update(aux_weight_dict)

    # add 3 keys (to `weight_dict`) for the intermdiate loss weights:
    #   loss_ce_interm loss_bbox_interm, loss_giou_interm
    # multiplies interm_loss_coef*_coeff_weight_dict*initial_loss_coefs to get the coeffs
    # but the default case is just 1.0 for both coeffs, so these new_key_vals=initial_key_vals
    if two_stage_type != "no":
        interm_weight_dict = {}

        no_interm_box_loss = False

        _coeff_weight_dict = {
            "loss_ce": 1.0,
            "loss_bbox": 1.0 if not no_interm_box_loss else 0.0,
            "loss_giou": 1.0 if not no_interm_box_loss else 0.0,
        }

        # default: 1.0
        interm_loss_coef = loss_args["interm_loss_coef"]

        interm_weight_dict.update(
            {
                k + f"_interm": v * interm_loss_coef * _coeff_weight_dict[k]
                for k, v in clean_weight_dict_wo_dn.items()
            }
        )
        weight_dict.update(interm_weight_dict)

        losses = [
            "labels",
            "boxes",
            "cardinality",
        ]  # NOTE: removing mask loss since unused

        # create the loss function used in dino and move to gpu
        criterion = SetCriterion(
            num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            focal_alpha=matcher_args["focal_alpha"],
            losses=losses,
        )
        criterion.to(device)

    return criterion
