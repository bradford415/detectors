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
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
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
            losses["class_error"] = (
                100 - topk_accuracy(src_logits[idx], target_classes_o)[0]
            )
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
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
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
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
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        device = next(iter(outputs.values())).device
        indices = self.matcher(outputs_without_aux, targets)

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs["dn_meta"]

        if self.training and dn_meta and "output_known_lbs_bboxes" in dn_meta:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    t = torch.range(0, len(targets[i]["labels"]) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (
                        torch.tensor(range(scalar)) * single_pad
                    ).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes = dn_meta["output_known_lbs_bboxes"]
            l_dict = {}
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
                        num_boxes * scalar,
                        **kwargs,
                    )
                )

            l_dict = {k + f"_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            l_dict = dict()
            l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["loss_ce_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["loss_xy_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["loss_hw_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["cardinality_error_dn"] = torch.as_tensor(0.0).to("cuda")
            losses.update(l_dict)

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for idx, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
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
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

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
                    l_dict = dict()
                    l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["loss_ce_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["loss_xy_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["loss_hw_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["cardinality_error_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict = {k + f"_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
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

        # enc output loss
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
        output_known_lbs_bboxes = dn_meta["output_known_lbs_bboxes"]
        num_dn_groups, pad_size = dn_meta["num_dn_group"], dn_meta["pad_size"]
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups


def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

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
