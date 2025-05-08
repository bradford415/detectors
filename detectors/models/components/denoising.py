# Denoising functions for DINO
import torch
from torch import nn


def setup_contrastive_denoising(
    training: bool,
    num_queries: int,
    num_classes: int,
    hidden_dim: int,
    label_enc: nn.Embedding,
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

        # Extract labels, boxes, and the image_index for all objects
        labels = torch.cat(
            [target["labels"] for target in targets]
        )  # (num_objects_in_batch,)
        boxes = torch.cat(
            [target["boxes"] for target in targets]
        )  # (num_objects_in_batch, 4)
        batch_idx = torch.cat(
            [
                torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)
            ]  # (num_objects_in_batch,)
        )

        # Create indices for each target (num_objects_in_batch,)
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        # repeat the indices for each denoise query (num_objects_in_batch*denoise_number*2,);
        # the *2 should be for pos & neg queries; above is also multiplied by 2 but it looks like
        # it gets canceled out, so this line is truly where it doubles the queries
        known_indice = known_indice.repeat(2 * denoise_number, 1).view(-1)

        # repeat labels and batch_idxs for each denoise query (num_objects_in_batch*denoise_number*2,)
        known_labels = labels.repeat(2 * denoise_number, 1).view(-1)
        known_batch_idx = batch_idx.repeat(2 * denoise_number, 1).view(-1)

        # repeat the the bboxes for each denoise query (num_objects_in_batch*denoise_number*2, 4)
        known_bboxs = boxes.repeat(2 * denoise_number, 1)

        # Create copies of the repeated known_labels and known_bbox
        known_labels_expand = known_labels.clone()  # (num_objects*denoise_number*2,)
        known_bbox_expand = known_bboxs.clone()  # (num_objects*denoise_number*2, 4)

        # inject random class labels into known_labels_expand (num_objects*denoise_number*2,)
        if denoise_label_noise_ratio > 0:
            # rand uniform [0, 1) tensor to represent the probability of choosing a noised label
            probs = torch.rand_like(
                known_labels_expand.float()
            )  # (num_objects_batch*denoise_number*2,)

            # Choose random denoise_queries w/ probability denoise_label_noise_ratio * 0.5 (default=0.25);
            # chosen_indice will be indices that were randomly chosen from probs; shape (num_chosen_indices,)
            chosen_indice = torch.nonzero(
                probs < (denoise_label_noise_ratio * 0.5)
            ).view(
                -1
            )  # half of bbox prob

            # Randomly choose a label between [0,num_classes); shape (num_chosen_indices,)
            new_label = torch.randint_like(
                chosen_indice, 0, num_classes
            )  # randomly put a new one here

            # inject random class labels into known_labels_expand; the random labels are copied from
            # new_label at locations chose_indice across dim 0; shape ((num_objects_batch*denoise_number*2,)
            known_labels_expand.scatter_(dim=0, index=chosen_indice, src=new_label)

        # num_objects for the image with the most objects in the batch
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * denoise_number)

        # The below code creates two variables: positive_idx, and negative idx;
        # the goal is to create indices of positive & negative denoising queries; these indices
        # will alternate in sets [0, num_objects_batch] until (num_queries_batch*denoise_number*2 - 1);
        # i.e., if num_queries_batch=19 then [0, 18] are positive indices, [19, 37] are negative queries,
        # and so on until the last index is 379 (num_queries_batch*denoise_number*2 - 1)

        # Create a range tensor from [0, num_objects_batch) and repeat along rows
        # (denoise_number, num_objects_batch)
        positive_idx = (
            torch.arange(boxes.shape[0])
            .long()
            .cuda()
            .unsqueeze(0)
            .repeat(denoise_number, 1)
        )

        # Create a range tensor from [0, denoise_number), mult by the num_objects_batch*2 (denoise_number, 1)
        # Sum the positive_idx and the newly created tensor, this broadcasts the new tensor along the cols
        positive_idx += (
            (torch.tensor(range(denoise_number)) * len(boxes) * 2)
            .long()
            .cuda()
            .unsqueeze(1)
        )

        # flatten to (denoise_number*num_objects_batch,)
        positive_idx = positive_idx.flatten()

        # Offset the positive_idx by the num_objects batch; this alternates the indices of the positive
        # and negative denoising quries by sets of num_objects_batch; e.g., if num_queries_batch=19 then
        # [0-18] = positive queries and [19-37] = negative queries and so this alternates until the last
        # index is (num_queries_batch*denoise_number*2 - 1) = 379
        negative_idx = positive_idx + len(boxes)

        # TODO: the default parameters of this var is 1.0 but config is 0.4; need to investigate
        if denoise_box_noise_scale > 0:
            # create 0s tensor as a placeholder (num_objects_batch*denoise_number*2, 4)
            known_bbox_x1y1x2y2 = torch.zeros_like(known_bboxs)

            # Convert (cx, cy, w, h) to (tl_x, tl_y, br_x, br_y)
            known_bbox_x1y1x2y2[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_x1y1x2y2[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            # create another 0s tensor as a placeholder (num_objects_batch*denoise_number*2, 4)
            diff = torch.zeros_like(known_bboxs)

            # Store half the w/h (w/2, h/2, w/2, h/2)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            # Create rand uniform ints -1 or 1 which represents the sign (num_objects_batch*denoise_number*2, 4)
            rand_sign = (
                torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32)
                * 2.0
                - 1.0
            )

            # Create rand uniform floats [0,1); shape (num_objects_batch*denoise_number*2, 4)
            rand_part = torch.rand_like(known_bboxs)

            # Add 1 to the `negative_idx` indices; reminder this alternates every num_class_batch intervals
            rand_part[negative_idx] += 1.0

            assert rand_part.shape == rand_sign.shape

            # Randomly flip the sign for all labels
            rand_part *= rand_sign

            # Apply random noise to the GT boxes (tl_x, tl_y, br_x, br_y) by adding an element-wise 
            # mult the rand noise [-2, 2] with the diff (w/2, h/2, w/2, h/2) and scale the values by 
            # denoise_box_noise_scale (default 1.0)
            known_bbox_x1y1x2y2 = (
                known_bbox_x1y1x2y2
                + torch.mul(rand_part, diff).cuda() * denoise_box_noise_scale
            )
            # bound [0, 1]
            known_bbox_x1y1x2y2 = known_bbox_x1y1x2y2.clamp(min=0.0, max=1.0) 
            
            # Convert the randomly noised boxes from (tl_x, tl_y, br_x, br_y) to (cx, cy, w, h);
            # NOTE: these will have some weird values
            known_bbox_expand[:, :2] = (known_bbox_x1y1x2y2[:, :2] + known_bbox_x1y1x2y2[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_x1y1x2y2[:, 2:] - known_bbox_x1y1x2y2[:, :2]
        
        # (num_objects_batch*denoise_number*2,)
        m = known_labels_expand.long().to("cuda")
        
        # Embed the object labels which contain the randomly injected labels 
        # (num_objects_batch*denoise_number*2, hidden_dim)
        input_label_embed = label_enc(m)
        
        ####### START HERE##############
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)        
            
        
