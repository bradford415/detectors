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
        # Double the number of denoise_queries to create positive and negative queries
        denoise_number *= 2

        # Create a list of tensors (num_objects,) filled with 1s for each sample
        known = [(torch.ones_like(target["labels"])).cuda() for target in targets]
        
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
