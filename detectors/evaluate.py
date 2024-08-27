from typing import Dict, List

import numpy as np
import torch

from detectors.utils.box_ops import box_iou_modified, bbox_iou_git


def get_batch_statistics(outputs: List[torch.Tensor], targets: List[Dict], iou_threshold) -> List:
    """Compute true positives, predicted scores and predicted labels per sample.
    This function must be used after non_max_suppression
    
    Args:
        outputs: Model predictions after non_max_suppression has been applied
                 shape of each list element (max_preds, 6) where 6 = (tl_x, tl_y, br_x, br_y, conf, cls)
        targets: Ground truth labels for the image; at minimum this must contain bbox coords [tl_x, tl_y, br_x, br_y] for
                 each object and a corresponding class label
        iou_threshold: IoU threshold required to qualify as detected; IoU must be greater than or equal
                       to this value

    Returns:
        A list of len(batch_size) where each element has a list containing:
            1. a numpy array of true_postives (num_preds,); 1 for true positive and 0 for false positive; num_preds is limited to 300 in nms
            2. a tensor of class confidences (num_preds,)
            3. a tensor of the pred labels (num_preds,)
    """
    batch_metrics = []
    
    # Loop through each batch
    for sample_i in range(len(outputs)):
        if outputs[sample_i] is None:
            continue
        
        # Extract a single sample prediction
        output = outputs[sample_i]

        # Extract the labels for the sample
        target = targets[sample_i]

        # Extract box preds (xyxy, objectness, class score)
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        # (max_nms_preds,); this is defined by max_nms in non_max_suppression()
        true_positives = np.zeros(pred_boxes.shape[0])

        #annotations = targets[targets[:, 0] == sample_i][:, 1:] # I think the code has its targets as [image_index, class_index, cx, cy, w, h], need to figure out if this is changed to xyxy and if i need to == sample
        annotations = torch.cat([torch.unsqueeze(target["labels"], 1), target["boxes"]], dim=1)
        target_labels = torch.unsqueeze(target["labels"], 1) if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            # Loop through each box prediction
            for pred_i, (pred_box, pred_label) in enumerate(
                zip(pred_boxes, pred_labels)
            ):
                # If all the targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                # Filter target_boxes by pred_label so that we only match against boxes of our own label
                # Explanation: I think x is a tuple of (index, tensor), where tensor is a single box coord
                #              from target_boxes; x[0] is the index; when target_labels at that index equals
                #              the pred_label then it will return the index and the target_box coord
                filtered_target_position, filtered_targets = zip(
                    *filter(
                        lambda x: target_labels[x[0]] == pred_label, 
                        enumerate(target_boxes),
                    )
                )
                # filtered_targets contains only the target boxes where the label matches the predicted label;
                # filtered_target_positions is the indices of the rows in target_boxes for the pred_label


                # iou, box_filtered_index = box_iou_modified(
                #     pred_box.unsqueeze(0), torch.stack(filtered_targets), return_union=False
                # ).max(0)
                # Find the best matching target for our predicted box;
                # predicted box is a single box prediction and filtered targets is 1 or more box labels
                # which allows us to try and match the predicted box with the best overlapping target
                iou, box_filtered_index = bbox_iou_git(
                    pred_box.unsqueeze(0), torch.stack(filtered_targets)
                ).max(0)

                # Remap the index in the list of filtered targets for that label to the index in the list with all targets.
                box_index = filtered_target_position[box_filtered_index]

                # Check if the iou is above the min threshold and i
                if iou >= iou_threshold and box_index not in detected_boxes:
                    # if detected, set the true_positives tensor to 1 for that prediction
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
        
    return batch_metrics
