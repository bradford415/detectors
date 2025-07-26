## DINO
### constrastive denoising (cdn) attention mask
The attention mask, `attn_mask`, created from `models.components.denoising.setup_for_cdn()`.
* white means to mask (block) attention and black means to attend

In this example:
* the maximum number of objects per image in the batch is `10`, there are `10` denoising queries per GT-object, and each denoising_query has a positive and negative query so `10*10*2 = 200`
* each `cdn_group` has `20` denoising queries so [0,200] represents the total number of denoising queries,
* the staircase pattern implies that only denoising queries within the same `cdn_group` can attend to each other (black square)
  * True = Block attention, False = Allow attention
* [200, 1100] represents learnable object queries and we want all of them to attend to eachother so we do not need to ask them out, hence why the right of the image is all black

![attn_mask](https://github.com/user-attachments/assets/04215378-04af-41e7-a630-4753731a6fce)


### loss components
Once dino computes the loss through `SetCriterion`, it returns many different loss components; this section briefly describes them and serves as a reference
* With the default parameters, `SetCriterion()` outputs a `loss_dict` with `79` components however only about `39` of them are used
* Each component is scaled by a value in the `weight_dict`, and only the loss components that appear in this `weight_dict` are used in the total loss, which is about `39`

The following loss components are used in the total loss:
| loss components  | description
|------------------|----------------
| loss_ce          | 
| loss_bbox        |
| loss_giou        |
| loss_ce_dn       |
| loss_bbox_dn     |
| loss_giou_dn     |
| loss_ce_0        |
| loss_bbox_0      |
| loss_giou_0      |
| loss_ce_dn_0     |
| loss_bbox_dn_0   |
| loss_giou_dn_0   |
| loss_ce_1        |
| loss_bbox_1      |
| loss_giou_1      |
| loss_ce_dn_1     |
| loss_bbox_dn_1   |
| loss_giou_dn_1   |
| loss_ce_2        |
| loss_bbox_2      |
| loss_giou_2      |
| loss_ce_dn_2     |
| loss_bbox_dn_2   |
| loss_giou_dn_2   |
| loss_ce_3        |
| loss_bbox_3      |
| loss_giou_3      |
| loss_ce_dn_3     |
| loss_bbox_dn_3   |
| loss_giou_dn_3   |
| loss_ce_4        |
| loss_bbox_4      |
| loss_giou_4      |
| loss_ce_dn_4     |
| loss_bbox_dn_4   |
| loss_giou_dn_4   |
| loss_ce_interm   |
| loss_bbox_interm |
| loss_giou_interm |
