## DINO
### constrastive denoising (cdn) attention mask
The attention mask, `attn_mask`, created from `models.components.denoising.setup_for_cdn()`.
* white means to mask (block) attention and black means to attend

In this example:
* the maximum number of objects per image in the batch is `10`, there are `10` denoising queries per GT-object, and each denoising_query has a positive and negative query so `10*10*2 = 200`
* each `cdn_group` has `20` denoising queries so [0,200] represents the total number of denoising queries,
* the staircase pattern implies that only denoising queries within the same `cdn_group` can attend to eachother (black square)
* [200, 1100] represents learnable object queries and we want all of them to attend to eachother so we do not need to ask them out, hence why the right of the image is all black

![attn_mask](https://github.com/user-attachments/assets/04215378-04af-41e7-a630-4753731a6fce)
