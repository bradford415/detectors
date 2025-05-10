# Denoising functions for DINO
import torch
from torch import nn

from detectors.utils.misc import inverse_sigmoid


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """TODO

    Section 3.3 and Figure 3 of DINO paper

    NOTE: this function should be called in DINO.forward()

    Args:
        training: training (True) or inference (False)
        num_queries: total number of learnable object queries; TODO: verify this
        num_classes: TODO
        hidden_dim: transformer hidden dimension; TODO: add a little more
        label_enc: an nn.Embedding layer to embed the denoising_queries containing GT classes with
                   randomly injected classes
        targets: dictionaries of labels for each image in the batch; see DINO.forward() for more info
        denoise_number: TODO
        denoise_label_noise_ratio: TODO
        denoise_box_noise_scale: TODO

    Return:
        TODO: explain all return values as a summary, this will be very helpful; put location of attn_mask visual
        1. input_query_label: a tensor with GT-truth classes and randomly selected classes injected at random
                              locations, this tensor was then embedded with nn.Embedding
                              shape (batch_size, max_objects_batch*denoise_number_per_cdn_group*2, hidden_dim)
        2. input_query_bbox: a tensor with all  GT bboxes noised (positive and negative denoising queries), these
                             bboxes are then converted to logits w/ the inverse_sigmoid;
                             (batch_size, max_objects_batch*denoise_number_per_cdn_group*2, 4)
                             where 4 = (cx, cy, w, h)
        3. attn_mask: an attention mask where False = attend and True = mask/block attention;
                      mask has shape (tgt_size, tgt_size) tgt_size=all_dn_queries + learnable object queries
                      the region of the mask attn_mask[:all_dn_queries, :all_dn_queries] (top_left)
                      is composed of CDN groups and each CDN group is only allowed to attend to itself,
                      therefore, the mask looks like stepsin the top left; to the right of the CDN groups
                      are learnable_obj_queries and these are free to attend to one another so the right
                      side of the mask is all False;
                      see detectors/models/README.md for a visual of this attn_mask
        4. denoise_meta: a dicionary which stores the computed values used to build the denoising queries
                         as metadata; has the following keys:
                            1. pad_size: total number of denoising_queries (default 200)
                            2. num_dn_group: the number of denoising_queries per CDN group
                                             (this should vary by batch)


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
            # new_label at locations chosen_indice across dim 0; shape (num_objects_batch*denoise_number*2,)
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
            known_bbox_expand[:, :2] = (
                known_bbox_x1y1x2y2[:, :2] + known_bbox_x1y1x2y2[:, 2:]
            ) / 2
            known_bbox_expand[:, 2:] = (
                known_bbox_x1y1x2y2[:, 2:] - known_bbox_x1y1x2y2[:, :2]
            )

        # (num_objects_batch*denoise_number*2,)
        m = known_labels_expand.long().to("cuda")

        # Embed the object labels which contain the randomly injected labels
        # (num_objects_batch*denoise_number*2, hidden_dim)
        input_label_embed = label_enc(m)

        # Convert the noised boxes to logits
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        # Create a zeros tensor of (pad_size, hidden_dim) & (pad_size, 4)
        # pad_size = single_pad * 2 * denoise_number and single_pad = num_objects in image with most objects
        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        # Repeat the zeros array for the entire batch
        # (batch_size, pad_size, hidden_dim) & (batch_size, pad_size, 4)
        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        # Create a map of indices on where to place the GT embeddings in the zeros tensor created above;
        # the indices will span up to (num_objects_in_img_with_most_objs*denoise_number*2) (default 200);
        # NOTE: input_label and bbox will be longer than map_known_indice rows, because it also has to go in a specific sample index
        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            # Create a 1d range tensor for the num_objects in each image (e.g., [0, 1, 2, 3, 0, 1, 2])
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num]
            )  # [1,2, 1,2,3]

            # Repeat this pattern counting up, off-setting by single_pad every iteration until
            # 2*denoise_number iters (num_objects_batch*denoise_number*2)
            map_known_indice = torch.cat(
                [map_known_indice + single_pad * i for i in range(2 * denoise_number)]
            ).long()

        # Fill the input queries with their label & bboxes embeddings at their correct sample,
        # label and bbox index (batch_size, pad_size, hidden_dim) & (batch_size, pad_size, 4)
        if len(known_batch_idx):
            input_query_label[(known_batch_idx.long(), map_known_indice)] = (
                input_label_embed
            )
            input_query_bbox[(known_batch_idx.long(), map_known_indice)] = (
                input_bbox_embed
            )

        # total number of target queries (denoising_queries + learnable object queries);
        # this is also the input length to the transformer decoder; TODO: verify this
        tgt_size = pad_size + num_queries

        # Create a blank attention mask where every element is False (tgt_size, tgt_size)
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0

        # Set the bottom-left of the mask to True (the right side is for learnable object queries);
        # True = masked, False = can attend;
        # The goal of the attention mask is so denoising_queries only attend to themselves;
        # "For all detection queries, block attention to denoising queries."
        #   1. Standard detection queries must not cheat by looking at ground-truth-derived noisy queries.
        #   2. Ensures the detection part of the model truly learns from scratch, not from GT hints.
        attn_mask[pad_size:, :pad_size] = True

        # Mask each group of denoising queries to prevent attending to other groups;
        # this ensures group-level isolation during attention;
        # during denoising training, multiple groups of denoising queries are constructed:
        #    Group 0: [pos, pos, ..., neg, neg] (length = 2 × num_objects)
        #    Group 1: ...
        #    ...
        #    Group N-1
        # Therefore, each group should only attend to itself and not others; NOTE: these groups are shown
        # visually in Figure 3 CDN groupX
        # NOTE: See detectors/models/README.md for a visual of what the attention mask looks like
        for cdn_group_i in range(denoise_number):  # (default 10)
            if cdn_group_i == 0:
                # First iteration only
                # Example: if single_pad=10 the first iter allows attention (False) between [0:20, 0:20]
                #          and blocks attn (True) [0:20, 20:200]
                # Reminder: single_pad = number of the most objects per image in the batch
                #           pad_size = the maximum total num_denoising_queries for all groups
                attn_mask[
                    single_pad * 2 * cdn_group_i : single_pad * 2 * (cdn_group_i + 1),
                    single_pad * 2 * (cdn_group_i + 1) : pad_size,
                ] = True
            if cdn_group_i == denoise_number - 1:
                # Last iteration
                # Mask only the row_vals before the cdn_group since there won't be any groups after
                # [: last_cdn_group:]
                attn_mask[
                    single_pad * 2 * cdn_group_i : single_pad * 2 * (cdn_group_i + 1),
                    : single_pad * cdn_group_i * 2,
                ] = True
            else:
                # For all iterations but the last
                # Mask the row_vals after the group [cdn_group_i: cdn_group_i+1, cdn_group_i+1: pad_size]
                attn_mask[
                    single_pad * 2 * cdn_group_i : single_pad * 2 * (cdn_group_i + 1),
                    single_pad * 2 * (cdn_group_i + 1) : pad_size,
                ] = True
                # Mask the row_vals at the beginning of the group [cdn_group_i: cdn_group_i+1, 0: cdn_group_i]
                attn_mask[
                    single_pad * 2 * cdn_group_i : single_pad * 2 * (cdn_group_i + 1),
                    : single_pad * 2 * cdn_group_i,
                ] = True

        # Store the computed values used to build the denoising queries as metadata
        denoise_meta = {
            "pad_size": pad_size,  # number of total denoising queries across all groups (denoise_number*2*max_objects_batch,)
            "num_dn_group": denoise_number,  # number of denoising queries per group (not mulitplied by 2 here)
        }
    else:
        # During inference, set all the return values to None
        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        denoise_meta = None

    return input_query_label, input_query_bbox, attn_mask, denoise_meta
