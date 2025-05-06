# Denoising functions for DINO
import torch


def setup_contrastive_denoising(
    training: bool,
    num_queries: int,
    num_classes: int,
    hidden_dim: int,
    label_enc,
    targets: list[dict],
    denoise_number: int,
    denoise_label_noise_ratio,
    denoise_box_noise_scale,
):
    """TODO

    Section 3.3 and Figure 3 of DINO paper

    NOTE: this function should be called in DINO.forward()

    Args:
        training: training (True) or inference (False)
        num_queries:
        num_classes:
        hidden_dim: transformer hidden dimension; TODO: add a little more
        label_enc:
        targets: dictionaries of labels for each image in the batch; see DINO.forward() for more info
        denoise_number:
        denoise_label_noise_ratio:
        denoise_box_noise_scale:
    """
    if training:
        # Double the number of denoise_queries to create positive and negative queries;
        #   - positive queries are slightly noised gt boxes & labels; the model is expected to recover
        #     the correct box & lable from this noise
        #   - negative queries are incorrect labels or heavily noised boxes that do not match any object
        #     and the model should not output any confident prediction for these
        denoise_number *= 2

        # Create a list of tensors (num_objects,) filled with 1s for each sample
        known = [(torch.ones_like(target["labels"])).cuda() for target in targets]

        batch_size = len(known)

        # list of num_objects in each image
        known_num = [sum(k) for k in known]

        # Adjust the denoise_number depending on the maxi number of objects for an image in the batch
        if int(max(known_num)) == 0:
            # if there's no objects in any image in the batch set dn number to 1
            denoise_number = 1
        else:
            if denoise_number >= 100:
                # evenly divide the denoise_number into the max_objects*2 in the batch;
                # i.e., each object gets at least denoise_number
                denoise_number = denoise_number // (
                    int(max(known_num) * 2)
                )  # I think * 2 is for pos & neg queries but it seems like it effectively cancels it out;
                   # a several lines down known_indice is multiplied by 2 which might actually double the 
                   # queries for pos & neg
            elif denoise_number < 1:
                denoise_number = 1

        # This case triggers if max(known_num)*2 > denoise_number
        if denoise_number == 0:
            denoise_number = 1

        # (num_objects_for_entire_batch,)
        unmask_bbox = unmask_label = torch.cat(known)

        # Extract labels, boxes, and the image_index for all objects (num_objects_in_batch,) 
        labels = torch.cat([target["labels"] for target in targets])
        boxes = torch.cat([target["boxes"] for target in targets])
        batch_idx = torch.cat(
            [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )

        # Create indices for each target (num_objects_in_batch,)
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        # repeat the indices for each denoise query (num_objects_in_batch*denoise_number*2,)
        # the *2 should be for pos & neg queries; above is also multiplied by 2 but it looks like
        # it gets canceled out, so this line is truly where it doubles the queries 
        known_indice = known_indice.repeat(2 * denoise_number, 1).view(-1)
        
        ######## START HERE
        
        known_labels = labels.repeat(2 * denoise_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * denoise_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * denoise_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()
