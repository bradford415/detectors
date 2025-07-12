# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import copy
import math
import random
from typing import Optional

import torch
from torch import Tensor, nn

from detectors.models.components.dino import gen_encoder_output_proposals
from detectors.models.layers.common import MLP, activation_map
from detectors.models.layers.positional import gen_sineembed_for_position
from detectors.models.ops.modules import MSDeformAttn
from detectors.utils.misc import RandomBoxPerturber, inverse_sigmoid


class DeformableTransformer(nn.Module):
    """Deformable Transformer module used in DINO

    Includes the TransformerEncoder and TransformerDecoder

    The deformable transformer and deformable attention was initially introduced in deformable-detr; DINO
    and other detr-like models modify this transformer module from the original

    TODO: Explain more
    """

    def __init__(
        self,
        d_model=256,
        num_heads=8,
        num_obj_queries=900,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_unicoder_layers=0,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=True,
        query_dim=4,
        num_patterns: int = 0,
        modulate_hw_attn=False,
        # for deformable encoder
        deformable_encoder: bool = True,
        deformable_decoder: bool = True,
        num_feature_levels: int = 1,
        enc_n_points=4,
        dec_n_points=4,
        use_deformable_box_attn=False,
        box_attn_type="roi_align",
        # init query
        learnable_tgt_init=False,
        decoder_query_perturber=None,
        add_channel_attention=False,
        add_pos_value=False,
        random_refpoints_xy=False,
        # two stage
        two_stage_type="standard",
        two_stage_pat_embed=0,
        two_stage_add_query_num=0,
        two_stage_learn_wh=False,
        two_stage_keep_all_tokens=False,
        # evo of #anchors
        dec_layer_number=None,
        rm_enc_query_scale=True,
        rm_dec_query_scale=True,
        rm_self_attn_layers=None,
        key_aware_type=None,
        # layer share
        layer_share_type: Optional[str] = None,
        # for detach
        rm_detach=None,
        decoder_sa_type="sa",
        module_seq=["sa", "ca", "ffn"],
        # for dn
        embed_init_tgt=False,
        use_detached_boxes_dec_out=False,
    ):
        """Initalize the deformable transformer module

        Args:
            TODO:
            d_model: hidden_dim of the the transformer
            num_heads:
            num_obj_queries: number of learnable object queries; this is the max num of objects
                             DINO can detect single image; each query predicts a class label
                             (including background/no_object) and a bbox; these are seperate from
                             dn_queries
            num_encoder_layers:
            num_decoder_layers:
            num_unicoder_layers:
            dim_feedforward:
            dropout: dropout value used for the encoder and decoder
            activation:
            normalize_before:
            return_intermediate_dec:
            query_dim:
            num_patterns:
            modulate_hw_attn: hardcoded set to True but never actually used
            deformable_encoder:
            deformable_decoder:
            num_feature_levels:
            enc_n_points:
            dec_n_points:
            use_deformable_box_attn:
            box_attn_type:
            learnable_tgt_init:
            decoder_query_perturber:
            add_channel_attention:
            add_pos_value:
            random_refpoints_xy:
            two_stage_type: supported values: ["no", "standard"]
            two_stage_pat_embed:
            two_stage_add_query_num:
            two_stage_learn_wh:
            two_stage_keep_all_tokens:
            dec_layer_number:
            rm_enc_query_scale:
            rm_dec_query_scale:
            key_aware_type
            layer_share_type: a string of which layers to share: "encoder", "decoder", "both";
                              if None do not share any layers; default None
            rm_detach:
            decoder_sa_type: the type of self-attention to use for the decoder; supported values ["sa", "ca_label", "ca_content"]
                             "sa" = standard self-attention; TODO: verify sa and explain the other two types
            module_seq:
            embed_init_tgt:
            use_detached_boxes_dec_out:
        """
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.deformable_encoder = deformable_encoder
        self.deformable_decoder = deformable_decoder
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_obj_queries = num_obj_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out

        assert query_dim == 4

        # DeformableEncoder requires multiscale feature_maps
        if num_feature_levels > 1:
            assert (
                deformable_encoder
            ), "only support deformable_encoder for num_feature_levels > 1"

        # Verify deformable encoder is enabled if deformable_box_attn is used; this is False by default
        if use_deformable_box_attn:
            assert (
                deformable_encoder or deformable_encoder
            )  # NOTE: I think this is a bug in the original code

        # NOTE: removing encoder/decoder layer share checks because an assertion in the
        #       source code requires it to be None
        assert layer_share_type is None

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ["sa", "ca_label", "ca_content"]

        # Instantiate the deformable transformer encoder layer (just the layer, not the full encoder)
        if deformable_encoder:
            encoder_layer = DeformableTransformerEncoderLayer(
                d_model,
                dim_feedforward,
                dropout,
                activation,
                num_feature_levels,
                num_heads,
                enc_n_points,
            )
        else:
            raise NotImplementedError

        assert two_stage_type in [
            "no",
            "standard",
        ], "unknown param {} of two_stage_type".format(two_stage_type)

        # Create the TransformerEncoder module
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            encoder_norm,
            d_model=d_model,
            num_obj_queries=num_obj_queries,
            deformable_encoder=deformable_encoder,
            enc_layer_share=False,
            two_stage_type=two_stage_type,
        )

        # Create the deformable transformer decoder layer
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            num_heads,
            dec_n_points,
            use_deformable_box_attn=use_deformable_box_attn,
            box_attn_type=box_attn_type,
            key_aware_type=key_aware_type,
            decoder_sa_type=decoder_sa_type,
            module_seq=module_seq,
        )

        # Initalize the transformer decoder; responsible for refining intial reference points
        # (anchor points) into predicted bboxes
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            query_dim=query_dim,
            modulate_hw_attn=modulate_hw_attn,
            num_feature_levels=num_feature_levels,
            deformable_decoder=deformable_decoder,
            decoder_query_perturber=decoder_query_perturber,
            dec_layer_number=dec_layer_number,
            rm_dec_query_scale=rm_dec_query_scale,
            dec_layer_share=False,
            use_detached_boxes_dec_out=use_detached_boxes_dec_out,
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.num_obj_queries = num_obj_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        # Create self.level_embed which is a trainable tensor (num_feature_leves, d_model)
        # this parameter's values will be initialized in self._init_parameters()
        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(
                    torch.Tensor(num_feature_levels, d_model)
                )
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt

        # create self.tgt_embed nn.Embedding layer (num_obj_queries, d_model)
        # and initalize weight with a normal_distribution
        if (two_stage_type != "no" and embed_init_tgt) or (two_stage_type == "no"):
            self.tgt_embed = nn.Embedding(self.num_obj_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        # assign two stage attributes; default values "standard", 0, 0, False, respectively
        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh

        # Create a linear and norm layer for anchor selection at the output of encoder;
        # used after the encoder output has its padded and invalid regions set to 0 in
        # gen_encoder_output_proposals
        if two_stage_type == "standard":
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)

            # Next 3 if statements are skipped by default
            if two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = nn.Parameter(
                    torch.Tensor(two_stage_pat_embed, d_model)
                )
                nn.init.normal_(self.pat_embed_for_2stage)

            # skipped
            if two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(self.two_stage_add_query_num, d_model)

            if two_stage_learn_wh:  # skipped
                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:  # default case
                self.two_stage_wh_embedding = None

        # skipped by default
        if two_stage_type == "no":
            self.init_ref_points(num_obj_queries)  # init self.refpoint_embed

        # Both these attributes are set in models.dino.DINO.__init__() in the two stage block;
        # they're used to encode the encoder output to class embeddings and bbox embeddings to select
        # the topk proposals for the decoder (this is called after the TransformerEncoder);
        # there's an option to share these layers with the decoder prediction heads but False by default
        # class_embed is of type Linear(hidden, num_classes) - for coco num_classes=91 not 80;
        # explanation for why 91 instead of 80: https://github.com/facebookresearch/detr/issues/23#issuecomment-636322576
        self.enc_out_class_embed = (
            None  # topk proposals will be chosen from these embeded values
        )

        # bbox_embed is a single 3 layer MLP with hidden_dims=256 and output_dim=4; there' an op
        self.enc_out_bbox_embed = None

        # evolution of anchors; skipped by default so can be ignored for now
        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            if self.two_stage_type != "no" or num_patterns == 0:
                assert (
                    dec_layer_number[0] == num_obj_queries
                ), f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_obj_queries})"
            else:
                assert (
                    dec_layer_number[0] == num_obj_queries * num_patterns
                ), f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_obj_queries}) * num_patterns({num_patterns})"

        self._init_parameters()

        # skipped by default
        self.rm_self_attn_layers = rm_self_attn_layers
        if rm_self_attn_layers is not None:
            print(
                "Removing the self-attn in {} decoder layers".format(
                    rm_self_attn_layers
                )
            )
            for lid, dec_layer in enumerate(self.decoder.layers):
                if lid in rm_self_attn_layers:
                    dec_layer.rm_self_attn_modules()

        # skipped by default
        self.rm_detach = rm_detach
        if self.rm_detach:
            assert isinstance(rm_detach, list)
            assert any([i in ["enc_ref", "enc_tgt", "dec"] for i in rm_detach])
        self.decoder.rm_detach = rm_detach

    def _init_parameters(self):
        """Initialize

        1. Initializes mulitdimenional parameters (like weight matrices but NOT biases)
        2. Initializes MSDeformAttn defined by MSDeformAttn._reset_parameters()
        3. Initializes self.level_embed from a normal distribution (level_embed, d_model)
        """
        # Apply xaiver_uniform init to all multi-dimensional parameters like weight matrices;
        # this will skip single dimensional parameters like biases
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Initalize the Multiscale Deformable Attention based on MSDeformAttn._reset_parameters()
        # function; this is different than the current function
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        # Initalize the values of the self.level_embed parameter with values drawn from a
        # normal distribution
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

        # this is skipped by default but the intent is to bias the simoid to predict small bboxes early on;
        # log(0.05 / (1 - 0.05))) = ~ -2.94 and sigmoid(-2.94) = ~ 0.05
        if self.two_stage_learn_wh:
            nn.init.constant_(
                self.two_stage_wh_embedding.weight, math.log(0.05 / (1 - 0.05))
            )

    def get_valid_ratio(self, mask: torch.Tensor):
        """Calculates the propportion of the image that is not padded (i.e., the valid part)
        in both the height and width; this is where the padding mask is False (no padding);
        the max ratio value will be 1.0 for images with the H/W not padded at all

        Args:
            mask: the binary mask for a feature map which indicates where real pixels are (False)
                  and where padded pixels are (True); shape (b, h, w)

        Returns:
            a tensor of width and height ratios for the batch which expresses what percentage
            of the width & height contains 'real' (valid) pixels (i.e., not padded);
            shape (b, 2) where first col is width_ratios and second col is height ratios
        """
        # Extract the first column and first row and count the number of 'real' pixels in
        # each batch; we only need the first row/column because of the way DETR pads
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)  # (b,)
        valid_W = torch.sum(~mask[:, 0, :], 1)  # (b,)

        # Calculate the percentage of the height & wdith that has 'real' (valid) pixels
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W

        # combine the width and height ratios for the batch; shape (b, 2) where the first column
        # is the width_ratios across the batch and the second column is the height_ratios across the batch
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2]
            )
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

    def forward(
        self,
        feature_maps: list[torch.Tensor],
        masks: list[torch.Tensor],
        pos_embeds: list[torch.Tensor],
        refpoint_embed,
        tgt,
        attn_mask=None,
    ) -> tuple[
        list,
        list,
    ]:
        """Call the DeformableTransformer

        Args:
            feature_maps: list of multiscale feature maps which were extracted from the backbone,
                          projected, and additional feature maps were created (b, h, w, c)
            masks: list of padding masks  (b, h, w)
            pos_embeds: list of positional embeds (b, hiddne_dim//2, h, w)

            Args returned from setup_contrastive_denoising()
            refpoint_embed: during training:
                                noised GT bboxes representing positive and negative denoising
                                queries; pos & neg queries alternate for all objects in the batch
                                denoise_number_per_cdn_group times; pos denoise queries are slightly noised
                                and expected to recover the GT object during prediction, while neg denoise
                                queries are heavily noised and expected to predict `no_object/background`
                                (batch_size, max_objects_batch*denoise_number_per_cdn_group*2, 4)
                                where 4 = (cx, cy, w, h);
                            during inference:
                                value is None
                            see setup_contrastive_denoising() return docs variable `input_query_bbox` for more info
            tgt: during training:
                    a tensor with GT-truth classes and randomly selected classes (from the enitre ontology)
                    injected at random locations, this tensor was then embedded with nn.Embedding;
                    approximately 25% of GT labels are randomly changed;
                    shape (batch_size, max_objects_batch*denoise_number_per_cdn_group*2, hidden_dim)
                 during inference:
                    value is None
            attn_mask: an attention mask where False = attend and True = mask/block attention;
                       mask has shape (tgt_size, tgt_size) tgt_size=all_dn_queries + learnable object queries
                       the region of the mask attn_mask[:all_dn_queries, :all_dn_queries] (top_left)
                       is composed of CDN groups and each CDN group is only allowed to attend to itself,
                       therefore, the mask looks like stepsin the top left; to the right of the CDN groups
                       are learnable_obj_queries and these are free to attend to one another so the right
                       side of the mask is all False;
                       see detectors/models/README.md for a visual of this attn_mask

        Returns:
            decoder outputs:
                hs: a list of raw intermediate decoder outputs (with LayerNorm applied) after
                    each decoder layer len=num_decoder_layers; shape (b, num_queries, hidden_dim);
                    length of the list is num_decoder_layers
                references: a list of the initial reference points and the refined reference points
                            from the deocder; the refined reference points are the predicted offsets;
                            the list is of length num_decoder_layers + 1 and each element has
                            shape (b, num_queries, 4)
            encoder outputs:
                hs_enc: topk encoder output features that had padding locations masked to 0
                        and linearly projected (1, b, topk, hidden_dim)
                ref_enc: sigmoid of the topk classes from the encoder `output_memory` which
                         was embedded to boxes though an MLP and output proposals added
                         (which are masked at padded and invalid locations with `inf`s)
                         (1, b, topk, 4)
            init_box_proposal: initial box proposals using the topk class embeds from the
                               generated output proposals (gen_encoder_output_proposals()) and
                               apply sigmoid() to each element; shape (b, topk, 4)
                               (this is technically not an encoder output because it doesn't use
                               any of the encoded feature values, just the shapes)

        """
        # Prepare the input for the encoder
        f_maps_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        # loop through each feature_map level
        for lvl, (f_map, mask, pos_embed) in enumerate(
            zip(feature_maps, masks, pos_embeds)
        ):
            # Store spatial sizes from each feature map
            bs, c, h, w = f_map.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            # flatten feature maps, masks, and positonal embeddings
            f_map = f_map.flatten(2).transpose(1, 2)  # (b, h*w, hidden_dim)
            mask = mask.flatten(1)  # (b, h*w)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # (b, h*w, hidden_dim)

            # add the pos_embedding to the level_embedding; (b, h*w, hidden_dim) + (1, 1, hidden_dim)
            # this will broadcast across the batch and flattened spatial dims
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed

            # store the flattened pos_embed+lvl_embed, feature_map, and mask
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            f_maps_flatten.append(f_map)
            mask_flatten.append(mask)

        # convert lists to tensors
        f_maps_flatten = torch.cat(f_maps_flatten, 1)  # (b, sum(h_w * w_i), hidden_dim)
        mask_flatten = torch.cat(mask_flatten, 1)  # (b, sum(h_w, w_i))
        lvl_pos_embed_flatten = torch.cat(
            lvl_pos_embed_flatten, 1
        )  # (b, sum(h_w * w_i), hidden_dim)

        # (num_feature_maps, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=f_maps_flatten.device
        )

        # Create a tensor of `start` indices where each flattened feature map begins;
        # (num_feature_maps,)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        # Find the proportion of `real` (not padded) height/width pixels of each image
        # in the batch (b, num_feature_maps, 2)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage; TODO comment maybe
        enc_topk_proposals = enc_refpoint_embed = None

        # Encode the features through the TransformerEncoder
        # `memory` (b, sum(h_i, w_i), hidden_dim) is the encoded `f_maps_flatten`
        # and `enc_*` are None by the default parameters
        memory, enc_intermediate_output, enc_intermediate_refpoints = self.encoder(
            f_maps_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            ref_token_index=enc_topk_proposals,  # bs, nq
            ref_token_coord=enc_refpoint_embed,  # bs, nq, 4
        )

        # input and output of encoder should have same shape
        assert memory.shape == f_maps_flatten.shape

        if self.two_stage_type == "standard":
            if self.two_stage_learn_wh:
                input_hw = self.two_stage_wh_embedding.weight[0]
            else:
                # set input_hw to None
                input_hw = None

            # mask out padded & invalid output_memory locations and generate
            # inital bbox anchors (b, sum(h_i * w_i), 4); these bbox anchors will have offset
            # predictions added to them (offsets found by passing enc output through an MLP);
            # see function docstrings for more info
            # NOTE: output proposals only uses the memory shapes and not the actual encoded values
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, input_hw
            )

            # linearly project and norm the masked, encoded features output shape is same shape
            # as input (b, sum(h_i * w_i), hidden_dim)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            # skipped by default
            if self.two_stage_pat_embed > 0:
                bs, nhw, _ = output_memory.shape
                # output_memory: bs, n, 256; self.pat_embed_for_2stage: k, 256
                output_memory = output_memory.repeat(1, self.two_stage_pat_embed, 1)
                _pats = self.pat_embed_for_2stage.repeat_interleave(nhw, 0)
                output_memory = output_memory + _pats
                output_proposals = output_proposals.repeat(
                    1, self.two_stage_pat_embed, 1
                )

            # skipped by default
            if self.two_stage_add_query_num > 0:
                assert refpoint_embed is not None
                output_memory = torch.cat((output_memory, tgt), dim=1)
                output_proposals = torch.cat((output_proposals, refpoint_embed), dim=1)

            # Embed the encoder output_memory into class embeddings through a Linear layer;
            # used to select topk features as described in section 3.4 in DINO paper;
            # (b, sum(h_i * w_i), num_classes) for coco num_classes=91
            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)

            # compute the initial bbox coordinates by predicting offsets from the encoder output
            # and adding them, element-wise, to the inital anchor boxes (output_proposals); this includes masking out
            # the paddedand invalid regions with "inf"
            # (b, sum(h_i * w_i), 4) where 4 = (cx, cy, w, h)
            enc_outputs_coord_unselected = (
                self.enc_out_bbox_embed(output_memory) + output_proposals
            )

            # assign top_k proposals to pass into the decoder; default 900
            topk = self.num_obj_queries

            # Select the topk proposal (feature) indices with the highest class value
            # (b, topk); described in section 3.4
            #   1. find the max class value for each encoded feature
            #   2. only select the topk indices with the highest class values
            #      (e.g., sum(hi * wi) ~ 10000 and only 900 are selected);
            topk_proposals = torch.topk(
                enc_outputs_class_unselected.max(-1)[0], topk, dim=1
            )[1]

            # gather reference boxes along the features `dim` from the topk_proposal indices selected
            # by the class_embedding above; these reference boxes are initial anchor points for the decoder;
            # to begin to refine into actual predicted boxes; the values of `index` are used to select the
            # `row` (dim 1) and the column index (dim 2) of `index` selects the values in `src` along the
            # columns; see this post for how torch.gather() works:
            #   https://stackoverflow.com/questions/50999977/what-does-gather-do-in-pytorch-in-layman-terms
            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected,
                dim=1,
                index=topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
            )  # reminder that these boxes had the inverse sigmoid applied (logits)

            # output shape is the same as `index` shape
            assert (
                refpoint_embed_undetach.shape
                == topk_proposals.unsqueeze(-1).repeat(1, 1, 4).shape
            )
            # detaches refpoint_embed_ from the computation graph, making it a constant;
            # we do this for several reasons:
            #   1. we detach the reference points so we have fixed anchors to guide
            #      decoder queries, this helps improve the stability of the training
            #   2. by detaching, we make sure gradients don’t flow back through these reference
            #      points, so the parts of the model that produced them aren’t adjusted based on
            #      losses computed downstream from THESE points; the encoder and other parameters
            #      that preceed this detachment are still updated through other means, like
            #      the encoder features that get passed into the decoder, these reference points
            #      are just one branch in the computational graph
            #   3. I think one way to think about this is that the encoder should focus on producing
            #      rich features to pass into the decoder, not focusing on initalizing the best
            #      reference anchors; the reference anchors are just a byproduct from the encoder;
            #      by detaching the reference anchors, the encoder will not be updated by the
            #      reference points gradients; if we did not detach, this could confuse the decoder
            #      on what it's trying to learn
            #   4. the goal is for the decoder to refine these anchor reference points, this is what
            #      one thing the decoder should learn to do; detaching them helps to provide a
            #      "clean slate" for the decoder's progressive refinement. It essentially treats
            #      the initial reference points as "proposals" that the decoder then refines,
            #      rather than forcing the encoder to directly optimize these initial proposals to
            #      be perfect.
            # NOTE: the MLP used to help create refpoint_embed_undetach only updates its parameters
            #       where `refpoint_embed_undetach`` is used for computations, not where `refpoint_embed_`
            #       is used since it was detached for the comp graph and prevents gradients from flowing
            #       back
            refpoint_embed_ = refpoint_embed_undetach.detach()

            # Gather initial box proposals using the topk class embeds from the generated output proposoals
            # gen_encoder_output_proposals() and apply sigmoid() to each element; output_proposals shape
            # (b, sum(h_i * w_i), 4),  init_box_proposal (b, topk, 4) where topk by default is 900
            init_box_proposal = torch.gather(
                output_proposals,
                dim=1,
                index=topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
            ).sigmoid()  # sigmoid

            # NOTE: at the end out gen_encoder_output_proposals(), the inverse sigmoid is applied to
            #       output_proposals but then in then gathering the init_box_proposal we apply sigmoid again;
            #       I'm guessing when we generate the refpoint_embed we added the unsigmoided output_prosals
            #       so we want to use logits (inverse_sigmoid), or it may be used downstream to

            # TODO: I'm not sure of the difference between init_box_proposal and refpoint_embed_

            assert init_box_proposal.shape == refpoint_embed_.shape

            # Gather the encoded features from the topk chosen from the class embed (b, topk, 256)
            tgt_undetach = torch.gather(
                output_memory,
                dim=1,
                index=topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model),
            )
            if self.embed_init_tgt:
                # Extract the tgt_embed weight matrix and reshape the matrix, repeat for the new batch dim,
                # and swap batch and obj_queries dim;
                # (900, hidden_dim) -> (900, 1, hidden_dim) -> (900, 2, hidden_dim) -> (2, 900, hidden_dim);
                # NOTE: this uses the embedding layer like a learnable parameter matrix,
                #       rather than an calling the embedding module (used to embed token indices)
                # NOTE: unlike expand, repeat copies the tensor data (new memory) and repeat is differentiable
                tgt_ = (
                    self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
                )
            else:
                tgt_ = tgt_undetach.detach()

            # combine the denoise queries with the top
            if refpoint_embed is not None:
                # Combine noised boxes with positive and negative queries from setup_contrastive_denoising()
                # with the detached reference box anchors which were created from the encoded features and
                # embedded with an MLP (+ output_proposals) (b, max_objects*num_cdn_group*2 + topk, 4) ~ (b, 1100, 4)
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)

                # Combine noised class labels (randomly selected labels) from setup_contrastive_denoising()
                # and the extracted tgt_embed weight matrix (b, num_dn_queries + topk, hidden_dim)
                # NOTE: I believe tgt is the learnable content queries & the GT+noise fed into the decoder
                #       in the dino paper Figure 2
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

        # skipped; I think this is previous generation detr case (like a lot of this code)
        elif self.two_stage_type == "no":
            tgt_ = (
                self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, d_model
            refpoint_embed_ = (
                self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, 4

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(
                    self.num_obj_queries, 1
                )  # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()

        else:
            raise NotImplementedError(
                "unknown two_stage_type {}".format(self.two_stage_type)
            )

        # Call the TransformerDecoder to refine the intial reference points (anchor points) into bbox
        # predictions;
        # pass the `memory` straight from the encoder, not the `output_memory` that was masked and projected
        # returns a list of raw intermediate decoder outputs after each decoder layer and a list of
        # the intial and refined reference points (box locations) after each decoder layer;
        # shape of each list element: hs (b, num_queries, hidden_dim), references (b, num_queries, 4)
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),  # (num_dn_queries + topk, b, hidden_dim)
            memory=memory.transpose(0, 1),  # (sum(h_i * w_i), b, hidden_dim);
            tgt_mask=attn_mask,
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(
                0, 1
            ),  # (sum(h_i * w_i), b, hidden_dim)
            refpoints_unsigmoid=refpoint_embed.transpose(
                0, 1
            ),  # (sum(h_i * w_i), b, 4)
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
        )
        assert (
            len(hs) == self.num_decoder_layers
            and len(references) == self.num_decoder_layers + 1
        )

        # slightly adjust encoder outputs
        if self.two_stage_type == "standard":

            if self.two_stage_keep_all_tokens:
                hs_enc = output_memory.unsqueeze(0)
                ref_enc = enc_outputs_coord_unselected.unsqueeze(0)
                init_box_proposal = output_proposals
            else:  # default case

                # topk encoder output features that had padding locations masked to 0
                # and linearly projected (1, b, topk, hidden_dim)
                hs_enc = tgt_undetach.unsqueeze(0)

                # sigmoid of the topk reference boxes from the encoder `output_memory` which
                # was embedded to boxes though an MLP and output proposals added
                # (which are masked at padded and invalid locations with `inf`s)
                # (1, b, topk, 4)
                ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None

        return hs, references, hs_enc, ref_enc, init_box_proposal
        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
        # ref_enc: sigmoid coordinates. \
        #           (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None


class DeformableTransformerEncoderLayer(nn.Module):
    """The deformable encoder layer for the transformer encoder used in DINO

    Performs deformable self-attention and a two-layer ffn

    This is almost identical to the original DETR encoder https://arxiv.org/pdf/2005.12872
    in figure 10

    This is just the layer, not the full Encoder
    """

    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        """Initalize the deformable transformer encoder layer

        Args:
            d_model:
            d_ffn:
            dropout: dropout value for the deform transformer encoder layer; default is
                     0.0
            activation: the activation function to use after the first linear layer in the ffn
            n_levels:
            n_heads:
            n_points:
        """
        super().__init__()
        # NOTE: removed MSDeformableBoxAttention because it was unused

        # Initalize the deformable attention module;
        # NOTE: this is a custom CUDA kernel w/ cpp code, it requires an NVIDIA GPU and a special install
        self.self_attn = MSDeformAttn(
            d_model, n_levels, n_heads, n_points
        )  # TODO: One day I need to look through this code

        # Initialize dropout and norm
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Create the feedforward network (ffn) of the encoder
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = activation_map[activation]()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # NOTE: removing channel attention because not used

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Adds `pos` embeddings to the `tensor` element-wise

        tensor and pos should be the same shape

        Args:
            tensor: the tensor sequence (b, sum(h_i * w_i), hidden_dim)
            pos: the positional embeddings to add to the (b, sum(h_i * w_i), hidden_dim)
        """
        assert tensor.shape == pos.shape

        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):

        src2 = self.linear1(src)
        src2 = self.activation(src2)
        src2 = self.dropout2(src2)
        src2 = self.linear2(src2)

        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        key_padding_mask=None,
    ):
        """Forward pass through the encoder layer

        Args:
            src: the input tensor to compute the attention of (b, sum(h_i * w_i), hidden_dim)
            pos: the positional embeddings to add to the input tensor sequence
                 (b, sum(h_i * w_i), hidden_dim)
            reference_points: TODO
            spatial_shape:TODO
            level_start_index: TODO
            key_padding_mask: TODO

        Returns:
            An encoded tensor of the same shape as the input `src`
        """
        assert src.shape == pos.shape

        # Add positional embeddings and compute self-attention with the `src` tensor
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Pass through the ffn (b, TODO, hidden_dim)
        src = self.forward_ffn(src)

        # NOTE: removing channel attention because not used

        return src


class DeformableTransformerDecoderLayer(nn.Module):
    """The deformable decoder layer for the transformer decoder used in DINO

    Performs self-attention -> deformable cross-attention -> and a two-layer ffn

    For the first decoder layer, self-attention is performed on the combined tensor of
    noised class labels and the queries weight matrix tensor; for the remaining decoder layers,
    self-attention is performed on the output of the previous decoder layer

    deformable cross attention is performed on the self-attended output (queries) and the
    raw encoded features from the encoder output (values); the same encoder output is used
    at every decoder layer as the values tensor

    two-layer ffn on the cross attended output

    """

    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        key_aware_type=None,
        decoder_sa_type="ca",
        module_seq=["sa", "ca", "ffn"],
    ):
        """Initalize the deformable transformer decoder layer

        Args:
            d_model: total dimension of the model (hidden_dim); default 256
            d_ffn: output dimesnion of the 1st linear layer
            dropout: dropout value for each attn module and after each linear layer; default 0.0
            activation: type of activation function to use after the first linear layer in the ffn
            n_levels: number of multiscale feature map levels; default 4
            n_heads: number of heads in deformable cross attn and regular self attn; default 8
            n_points: TODO
            key_aware_type:
            decoder_sa_type: the type of self-attention to use in the decoder; default is "sa" which
                             uses a standard multiheaded self-attention module, not deformable attn
            module_seq: the order to call the modules in for the decoder forward() call; the order
                        must be ["sa", "ca", "ffn"] which stands for
                        ["self-attention", "cross-attention", "feedforward network"]; NOTE: this follows
                        the original DETR decoder very closely (see the DETR paper figure 10)
        """
        super().__init__()
        self.module_seq = module_seq
        assert sorted(module_seq) == ["ca", "ffn", "sa"]

        # NOTE: removed MSDeformableBoxAttention because it was unused

        # Create the cross-attn multiscale deformable attention module
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Create a regular multiheaded self-attention module (not deformable)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Create a 2 layer ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = activation_map(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ["sa", "ca_label", "ca_content"]

        if decoder_sa_type == "ca_content":
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Adds `pos` embeddings to the `tensor` element-wise

        tensor and pos should be the same shape

        Args:
            tensor: the tensor sequence (b, sum(h_i * w_i), hidden_dim)
            pos: the positional embeddings to add to the (b, sum(h_i * w_i), hidden_dim)
        """
        assert tensor.shape == pos.shape

        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """Forward pass through the ffn; the final module in the decoder layer

        Args:
            tgt: the tensor output from the cross-attention module forward_ca()
                  (num_queries, b, hidden_dim)

        Returns:
            returns the ffn output of the same shape as the input `tgt`
            (num_queries, b, hidden_dim)
        """
        # Linear project the tgt through a 2-layer ffn with dropout and activation
        tgt2 = self.dropout3(
            self.activation(self.linear1(tgt))
        )  # (num_queries, b, d_ffn)
        tgt2 = self.linear2(tgt)  # (num_queries, b, d_model)

        # Add the residual and layer normalize
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        """Perform regular multiheaded self-attention on the tgt (this is not deformable attention)

        this also adds the residual and layer normalization to the self-attn output

        NOTE: positional embeddings are only added to the query and key tensors,
              not the value tensor (this is how DETR does it)

        Args:
            see forward() docstring for the args

        Returns:
            The self-attended tgt tensor of the same shape as the input `tgt`
            (num_queries, batch_size, hidden_dim)
        """
        # self attention
        if self.self_attn is not None:
            if self.decoder_sa_type == "sa":
                # Add positional embeddings to the tgt; tgt_query_pos the projected query
                # (reference_point) positional embeddings created in gen_sineembed_for_position()
                q = k = self.with_pos_embed(tgt, tgt_query_pos)

                # perform self-attention on the tgt with the positional embeddings
                # NOTE that positional embeddings were only added to and q, k;
                #      v contains the actual contnt so if we add positionals it could
                #      interfere with the content; I don't think all self-attention
                #      implementations do this but DETR does
                tgt2 = self.self_attn(
                    query=q, key=k, value=tgt, attn_mask=self_attn_mask
                )[0]

                # Add the residual to the self-attn output and layer normalize;
                # dropout p=0.0 by default
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            # skipped by default
            elif self.decoder_sa_type == "ca_label":
                bs = tgt.shape[1]
                k = v = self.label_embedding.weight[:, None, :].repeat(1, bs, 1)
                tgt2 = self.self_attn(tgt, k, v, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == "ca_content":
                tgt2 = self.self_attn(
                    self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                    tgt_reference_points.transpose(0, 1).contiguous(),
                    memory.transpose(0, 1),
                    memory_spatial_shapes,
                    memory_level_start_index,
                    memory_key_padding_mask,
                ).transpose(0, 1)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            else:
                raise NotImplementedError(
                    "Unknown decoder_sa_type {}".format(self.decoder_sa_type)
                )

        return tgt

    def forward_ca(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        """Perform deformable multiheaded cross-attention on the tgt


        NOTE: deformable attention does not have an explicit `key` tensor

        NOTE: positional embeddings are only added to the query tensor,
              not the value tensor (this is how DETR does it)

        Args:
            tgt: the output from the multiheaded self-attn module (forward_sa()); these are
                 the queries
            tgt_reference_points: the reference points scaled by valid_ratios
                                  (num_queries, b, num_levels, 4) where 4 = (x, y, w, h)
            memory: raw encoded features directly from the output of the TransformerEncoder
                    (sum(h_i * w_i), b, hidden_dim); these are the values
            see forward() docstring for the remaining args

        Returns:
            the cross-attended tgt with `memory` (raw encoder output)
            with a residual added and layer normalized (num_queries, b, hidden_dim)
        """
        # cross attention
        # skipped by default
        if self.key_aware_type is not None:
            if self.key_aware_type == "mean":
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == "proj_mean":
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError(
                    "Unknown key_aware_type: {}".format(self.key_aware_type)
                )

        # Perform deformable cross-attention on the tgt with the memory features
        # (b, num_queries, hidden_dim) and then transposed (num_queries, b, hidden_dim)
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos).transpose(
                0, 1
            ),  # (b, num_queries, hidden_dim)
            reference_points=tgt_reference_points.transpose(
                0, 1
            ).contiguous(),  # (b, num_queries, num_levels, 4)
            input_flatten=memory.transpose(0, 1),  # (b, sum(h_i * w_i), hidden_dim)
            input_spatial_shapes=memory_spatial_shapes,
            input_level_start_index=memory_level_start_index,
            input_padding_mask=memory_key_padding_mask,
        ).transpose(0, 1)

        # add a residual and layer normalize; dropout p=0.0 by default
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        return tgt

    def forward(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        """Forward pass through the deformable transformer decoder layer

        Performs self-attention, cross-attention, and a two-layer ffn for every decoder layer

        Args:
            tgt: for the first decoder layer:
                    the combined noised class labels (randomly selected labels) from
                    setup_contrastive_denoising() and the extracted tgt_embed weight matrix
                    (num_queries, b, hidden_dim) where num_queries = max_objects*num_cdn_group*2 + topk
                    NOTE: num_queries = num_cdn_group*max_objects*2 + topk ~ 200 + 900
                          and I think these topk queries are the blue learnable content queries and
                          the cdn_group are the yellow/brown GT+noise queries fed into the decoder
                          in the DINO paper Figure 2
                 for the remaining decoder layers:
                    the output from the previous decoder layer
                    TODO: put shape
            tgt_query_pos: the projected query (reference_point) positional embeddings
                           (num_queries, b, hidden_dim)
            tgt_query_sine_embed: the raw query (reference_point) positional embeddings
                           (num_queries, b, hidden_dim*2); these ones are not projected with
                           the MLP like tgt_query_pos are
            tgt_key_padding_mask: None; unused in TransformerDecoder()
            tgt_reference_points: the reference points scaled by valid_ratios
                                  (num_queries, b, num_levels, 4) where 4 = (x, y, w, h)
            memory: raw encoded features directly from the output of the TransformerEncoder
                    (sum(h_i * w_i), b, hidden_dim); no post processing was done like `output_memory`
            memory_key_padding_mask: the flattened padding mask which expresses which pixels were padded
                                     in the input where True=padded and False=real_pixel
                                     (b, sum(h_w, w_i)); sum(h_w, w_i) = the flattened feature_map dim
            memory_level_start_index: start index of the level in sum(h_i * w_i) shape (num_levels,);
                                      e.g., the 1st level will start at index 0, the 2nd level will
                                      start on index feature_map[0]_h * feature_map[0]_w, etc..
                                      because the 2nd dim of src is flattened across all feature_maps
            memory_spatial_shapes: height and width of each feature_map level (num_level, 2); no batch
                                   dimension bc these values should be the same across the batch
            memory_pos: the flattened positional embeddings created in the Joiner()
                        (sum(h_i * w_i), b, hidden_dim);
                        NOTE: these positionals were added at the start of each encoder layer
            self_attn_mask: an attention mask where False = attend and True = mask/block attention
                            with shape (num_queries, num_queries);
                            see the `attn_mask` return value in setup_contrastive_denoising() for a
                            longer description;
                            also see detectors/models/README.md for a visual of this attn_mask
            cross_attn_mask: None; unused in TransformerDecoder()

        Returns:
            the output of the decoder layer after performing self-attention, cross-attention,
            and a two-layer ffn; the output is of the same shape as the input `tgt`
            (num_queries, batch_size, hidden_dim)
        """
        assert self.module_seq == ["sa", "ca", "ffn"]

        # Call each decoder module in the order specified by self.module_seq
        # reminder: the module or is ["sa", "ca", "ffn"] which stands for
        #           ["self-attention", "cross-attention", "feedforward network"];
        #           NOTE: this follows the original DETR decoder very closely (see the DETR paper figure 10)
        for funcname in self.module_seq:
            if funcname == "ffn":
                tgt = self.forward_ffn(tgt)
            elif funcname == "ca":
                tgt = self.forward_ca(
                    tgt,
                    tgt_query_pos,
                    tgt_query_sine_embed,
                    tgt_key_padding_mask,
                    tgt_reference_points,
                    memory,
                    memory_key_padding_mask,
                    memory_level_start_index,
                    memory_spatial_shapes,
                    memory_pos,
                    self_attn_mask,
                    cross_attn_mask,
                )
            elif funcname == "sa":
                tgt = self.forward_sa(
                    tgt,
                    tgt_query_pos,
                    tgt_query_sine_embed,
                    tgt_key_padding_mask,
                    tgt_reference_points,
                    memory,
                    memory_key_padding_mask,
                    memory_level_start_index,
                    memory_spatial_shapes,
                    memory_pos,
                    self_attn_mask,
                    cross_attn_mask,
                )
            else:
                raise ValueError("unknown funcname {}".format(funcname))

        return tgt


class TransformerEncoder(nn.Module):
    """The transformer encoder used in DINO"""

    def __init__(
        self,
        encoder_layer: DeformableTransformerEncoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        d_model: int = 256,
        num_obj_queries: int = 900,
        deformable_encoder: bool = False,
        enc_layer_share: bool = False,
        enc_layer_dropout_prob=None,
        two_stage_type: str = "standard",
    ):
        """Initalize the TransformerEncoder

        Args:
            encoder_layer:
            num_layers: number of encoder layers to stack; these will be looped through
                        sequentially in forward
            norm: Normalization module to use after at the end of the Encoder; by default
                  this is None, so no normalization is applied at the end
            d_model:
            num_obj_queries: number of learnable queries which predict a class label and a bounding box
            deformable_encoder:
            enc_layer_share: Whether to share all the encoder layers' parameters
                             (i.e., only one module is used); default is False
            enc_layer_dropout_prob: probablity
            two_stage_type:  supported values: ["no", "standard"]
        """
        super().__init__()
        # Create a list of new encoder layers which has len() num_layers;
        # the default case is to not share parameters between layers i.e., create deep copies
        if num_layers > 0:
            self.layers = _get_clones(
                encoder_layer, num_layers, layer_share=enc_layer_share
            )
        else:
            self.layers = []
            del encoder_layer

        assert len(self.layers) == num_layers

        self.query_scale = None
        self.num_obj_queries = num_obj_queries
        self.deformable_encoder = deformable_encoder
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model

        # setup the layer dropout probability if not None; this randomly skips
        # encoder layers; by default this is None and can be ignored
        self.enc_layer_dropout_prob = enc_layer_dropout_prob
        if enc_layer_dropout_prob is not None:
            assert isinstance(enc_layer_dropout_prob, list)
            assert len(enc_layer_dropout_prob) == num_layers
            for i in enc_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        #
        self.two_stage_type = two_stage_type

        # NOTE: removing the two_stage_setup  for ["enceachlayer", "enclayer1"] as they do
        #       not appear to be support

    @staticmethod
    def get_reference_points(
        spatial_shapes: torch.Tensor, valid_ratios: torch.Tensor, device: torch.device
    ):
        """Computes normalized reference points used by the deformable attention module in
        the encoder and decoder

        Reference points are the starting points for where attention should be computed,
        then an `offset` parameter is learned which will offset these reference points to
        where the model thinks the most important features to attend to are

        Args:
            spatial_shapes: height and width of each feature_map level (num_level, 2); no batch
                            dimension bc these values should be the same across the batch
            valid_ratios: a tensor of width and height ratios for each feature_map across the batch
                          which expresses what percentage of the width & height contains 'real' (valid)
                          pixels (i.e., not padded); shape (b, num_levels, 2) where 2 = width_ratios and height_ratios
                          and 4 is the number of levels (num_feature_maps); num_levels is typically 4
            device: the device to perform computations on

        Returns:
            x, y reference points which were normalized by the `real` pixel ratio
            (i.e., values > 1.0 are padded, not real), each reference point is scaled
            by the valid_ratio for each level (b, sum(h_i*w_i), num_levels, 2)
        """
        # used to store the reference points in an xy tensor for each level (each feature_map);
        # each element will be (b, h_i*w_i, 2)
        reference_points_list = []

        # for each feature map in the batch
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # initalize meshgrid ranging [0.5, height - 0.5] rows and [0.5, width - 0.5] cols
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing="ij",  # i->rows, j->cols
            )

            # normalize the reference points and bound them to the `real` region by flattening the
            # ref points (1, h*w) and divide by the `real` pixel height and width; valid_ratios multiplied
            # by the spatial dims gives the number of real pixels; (1, h*w) / (b, 1) = (b, h*w);
            # values > 1.0 are invalid
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)

            # combine x & y ref points and append to list (b, h*w, 2)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)

        # list length should be the number of feature maps
        assert len(reference_points_list) == spatial_shapes.shape[0]

        # combine all the reference points across all features map levels into a single
        # tensor (b, sum(h_i*w_i), 2) where 2 = (ref_x, ref_y)
        reference_points = torch.cat(reference_points_list, 1)

        # Scale and broadcast every reference point by the valid ratio for each feature map level
        # (b, sum(h_i*w_i), 1, 2) * (b, 1, num_levels, 2) = (b, sum(h_i*w_i), num_levels, 2);
        # I'm entirely sure why we broadcast each ref point with the ratios for each level,
        # since 2nd dim already includes these (concatenated together in the previous line)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        src: Tensor,
        pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        key_padding_mask: Tensor,
        ref_token_index: Optional[Tensor] = None,
        ref_token_coord: Optional[Tensor] = None,
    ):
        """Call the TransformerEncoder

        The main goal is to loop through the stack of encoders and encode the
        input features

        Args:
            NOTE: h_i & w_i -> height and width of a feature_map from the list of feature_maps
            src: the input tensor to compute the attention of (b, sum(h_i * w_i), hidden_dim)
            pos: the positional embeddings to add to the input tensor sequence
                 (b, sum(h_i * w_i), hidden_dim); it appears that postional are added
                 at every encoder layer and not just once before the encoder starts
            spatial_shapes: height and width of each feature_map level (num_level, 2)
            level_start_index: start point of level in sum(h_i * w_i) (num_level,);
                               e.g., the 1st level will start at index 0, the 2nd level will
                               start on index feature_map[0]_h * feature_map[0]_w, etc..
                               because the 2nd dim of src is flattened across all feature_maps
            valid_ratios: TODO (b, num_level, 2)
            key_padding_mask: TODO[bs, sum(hi*wi)]
            ref_token_index: TODO bs, nq
            ref_token_coord: TODO bs, nq, 4
            reference_points: TODO [bs, sum(hi*wi), num_level, 2]

        Returns:
            output: the enhanced features after being propagated through the stack
                    of transformer encoder layers (b, sum(h_i * w__), hidden_dim)
                    where hidden_dim typically is 256
            NOTE: the other 2 return values are empty lists due to the default parameters
        """
        if self.two_stage_type in ["no", "standard"]:
            assert ref_token_index is None

        output = src

        # Create reference points which act as the starting point for deformable attention;
        # deformable attention then learns `offsets` which shifts the location of the reference points
        # to where the model thinks it should attend; these `reference_points` were normalized
        # by the feature_map h/w of only the 'valid' regions (i.e., no padding), not the full h/w
        if self.num_layers > 0:
            if self.deformable_encoder:
                # (b, sum(h_i*w_i), num_levels, 2)
                reference_points = self.get_reference_points(
                    spatial_shapes, valid_ratios, device=src.device
                )

        # TODO
        intermediate_output = []
        intermediate_ref = []

        # is None by default so we'll skip this
        if ref_token_index is not None:
            out_i = torch.gather(
                output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model)
            )
            intermediate_output.append(out_i)
            intermediate_ref.append(ref_token_coord)

        # main process; loop through the list of DeformableTransformerEncoderLayers
        # and pass the output of the current layer into the next layer
        for layer_id, layer in enumerate(self.layers):
            # main process
            dropflag = False

            # NOTE: removing the layer dropping becuase it is unused

            # forward pass through the current encoder layer
            if not dropflag:
                if self.deformable_encoder:
                    output = layer(
                        src=output,  # see `src` in the docstring for `output` description
                        pos=pos,  # pos embeddings are added at every layer
                        reference_points=reference_points,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        key_padding_mask=key_padding_mask,
                    )
                else:
                    output = layer(
                        src=output.transpose(0, 1),
                        pos=pos.transpose(0, 1),
                        key_padding_mask=key_padding_mask,
                    ).transpose(0, 1)

            # NOTE: removing unsupported two_stage_type (['enceachlayer', 'enclayer1'])
            #       if statement

            # NOTE: also removing the aux loss if statement since it never gets used
            #       in the encoder (ref_token_index is None)

        # pre_norm by default is False so this is skipped
        if self.norm is not None:
            output = self.norm(output)

        if ref_token_index is not None:
            intermediate_output = torch.stack(
                intermediate_output
            )  # n_enc/n_enc-1, bs, \sum{hw}, d_model
            intermediate_ref = torch.stack(intermediate_ref)
        else:
            intermediate_output = intermediate_ref = None

        return output, intermediate_output, intermediate_ref


class TransformerDecoder(nn.Module):
    """The transformer decoder used in DINO

    The ultimate goal of the Decoder is to refine the intial reference points (anchor points)
    by predicting offset corrections (`outputs_unsig = delta_unsig + reference_before_sigmoid`)
    at every decoder layer; these refined `reference_points` are then passed to the next decoder
    layer

    reference points are like bbox locations of where objects might be (cx, cy, w, h)

    """

    def __init__(
        self,
        decoder_layer: DeformableTransformerDecoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        return_intermediate: bool = False,
        d_model: int = 256,
        query_dim: int = 4,
        modulate_hw_attn: bool = False,
        num_feature_levels: int = 1,
        deformable_decoder: bool = False,
        decoder_query_perturber=None,
        dec_layer_number=None,  # number of queries each layer in decoder
        rm_dec_query_scale=False,
        dec_layer_share: bool = False,
        dec_layer_dropout_prob: Optional[bool] = None,
        use_detached_boxes_dec_out=False,
    ):
        """Initalize the transformer decoder

        Args:
            decoder_layer: deformable transformer decoder layer,
            num_layers: number of decoder layers to stack; these will be looped through
                        sequentially in forward
            norm: the type of normalization to use; default nn.LayerNorm
            return_intermediate: whether to return the intermediate layers from the decoder; default is True
                                 and this parameter isn't really even used, it just throws an assert if False
            d_model: total dimension of the transformer decoder; this dim will be split across
            query_dim: TODO
            modulate_hw_attn:
            num_feature_levels:
            deformable_decoder:
            decoder_query_perturber:
            dec_layer_number: TODO  # number of queries each layer in decoder
            rm_dec_query_scale: TODO,
            dec_layer_share: whether to use the same decoder layer and share parameters; default False,
                             do not share
            dec_layer_dropout_prob: by default is None and can be ignored
            use_detached_boxes_dec_out=False,

        """
        super().__init__()

        # Create a list of new decoder layers which has len() num_layers;
        # the default case is to not share parameters between layers i.e., create deep copies
        if num_layers > 0:
            self.layers = _get_clones(
                decoder_layer, num_layers, layer_share=dec_layer_share
            )
        else:
            self.layers = []

        assert len(self.layers) == num_layers

        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"

        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)

        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out

        # a 2 layer MLP used to embed the reference point positional embeddings
        self.ref_point_head = MLP(
            input_dim=query_dim // 2 * d_model,  # default 512
            hidden_dim=d_model,
            output_dim=d_model,
            num_layers=2,
        )

        # NOTE: removing if statement if deformable_decoder is False since it's always True

        # TODO: comment
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:  # default case
            self.query_pos_sine_scale = None

        self.query_scale = None

        # this will be set in DINO.__init__() and is a module list of 3-layer MLP modules
        # to embed the bboxes after each decoder layer; the number of MLP modules is the number
        # of decoder layers; by default these share parameters like in the original DETR
        self.bbox_embed = None

        # set in DINO.__init__(); ModuleList of the class prediction module for the output of
        # each decoder layer; by default these share parameters
        self.class_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:  # default case
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number

        # default is None and can be ignored for now
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers

        self.dec_layer_dropout_prob = dec_layer_dropout_prob

        # default is None and can be ignored from now
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.rm_detach = None

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
        # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
        spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        valid_ratios: Optional[Tensor] = None,
    ):
        """TODO

        Args:
            NOTE: several of the parameters have their shape transposed upon input
            tgt: combined noised class labels (randomly selected labels) from setup_contrastive_denoising
                 and the extracted tgt_embed weight matrix
                 (num_queries, b, hidden_dim)
                NOTE: num_queries = num_cdn_group*max_objects*2 + topk ~ 200 + 900
                      and I think these topk queries are the blue learnable content queries and
                      the cdn_group are the yellow/brown GT+noise queries fed into the decoder
                      in the DINO paper Figure 2
            memory: raw encoded features directly from the output of the TransformerEncoder
                    (sum(h_i * w_i), b, hidden_dim); no post processing was done like `output_memory`
            tgt_mask: an attention mask where False = attend and True = mask/block attention;
                      see the `attn_mask` return value in
                      models.components.dino.setup_contrastive_denoising() for a longer description;
                      also see detectors/models/README.md for a visual of this attn_mask
            memory_mask: unused
            tgt_key_padding_mask: unused
            memory_key_padding_mask: the flattened padding mask which expresses which pixels were padded
                                     in the input where True=padded and False=real_pixel
                                     (b, sum(h_w, w_i)); sum(h_w, w_i) = the flattened feature_map dim
            pos: the flattened positional embeddings (sum(h_i * w_i), b, hidden_dim);
                 NOTE: these positionals were added at the start of each encoder layer
            refpoints_unsigmoid: combined noised boxes with positive and negative queries
                                 from setup_contrastive_denoising() with the detached reference box
                                 anchors which were created from the encoded features and embedded
                                 with an MLP (+ output_proposals); these will be used as initial
                                 reference points, the decoder_layer will predicted and refine offsets
                                 and these offsets will be used to predict the bboxes from these
                                 initial reference points;
                                 shape (max_objects*num_cdn_group*2 + topk, b, 4) ~ (b, 1100, 4)
                                 where 4 = (cx, cy, w, h)
            level_start_index: start index of the level in sum(h_i * w_i) shape (num_levels,);
                               e.g., the 1st level will start at index 0, the 2nd level will
                               start on index feature_map[0]_h * feature_map[0]_w, etc..
                               because the 2nd dim of src is flattened across all feature_maps
            spatial_shapes: height and width of each feature_map level (num_level, 2); no batch
                            dimension bc these values should be the same across the batch
            valid_ratios: a tensor of width and height ratios for each feature_map across the batch
                          which expresses what percentage of the width & height contains 'real' (valid)
                          pixels (i.e., not padded); shape (b, num_levels, 2) where 2 = width_ratios and height_ratios
                          and 4 is the number of levels (num_feature_maps); num_levels is typically 4;
                          a ratio of 1.0 means the H or W dimension has no padding
                          (1.0 is also the highest it can be)

        Returns:
            a two element list of
                1. a list of raw intermediate decoder outputs (with LayerNorm applied) after
                   each decoder layer len=num_decoder_layers; each element is transposed for
                   shape (b, num_queries, hidden_dim)
                2. a list of the initial reference points and the refined reference points;
                   the refined reference points are the predicted offsets + the reference points
                   from the previous layer; the list is of length num_decoder_layers + 1 and
                   each element is tranposed for shape (b, num_queries, 4)
        """
        output = tgt

        intermediate = []

        # bound reference points between [0,1]
        reference_points = refpoints_unsigmoid.sigmoid()

        # save the initial reference_points in a list; later on after each decoder layer
        # new reference points will be appended here
        ref_points = [reference_points]

        # Loop through each DeformableTransformerDecoderLayer and refine the intial reference
        # points to create bbox offset predictions; default 6 decoder layers
        for layer_id, layer in enumerate(self.layers):

            # skipped by default (decoder_query_perturber=None)
            if (
                self.training
                and self.decoder_query_perturber is not None
                and layer_id != 0
            ):
                reference_points = self.decoder_query_perturber(reference_points)

            if self.deformable_decoder:

                # NOTE: these `reference_points` are refined at every decoder layer and the refined
                #       `reference_points` will be the input to the next decoder layer
                if (
                    reference_points.shape[-1] == 4
                ):  # if reference points are in cxcywh format (default case)

                    # Scale the ref boxes by the valid_ratios (proportion of image that is not padded)
                    # create new singles dim and concat the valid ratios along the last dim
                    # (num_queries, b, 1, 4) * (1, b, num_levels, 4) = (num_queries, b, num_levels, 4)
                    reference_points_input = (
                        reference_points[:, :, None]
                        * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                    )  # nq, bs, nlevel, 4
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = (
                        reference_points[:, :, None] * valid_ratios[None, :]
                    )

                # generate sinusoidal positional embeddings for the reference points (num_queries, b, 512);
                # these are slightly different than the pos embeddings generated for the f_maps
                # in the sense that the temperature parameter is 10000 (like in DETR) as opposed to
                # 40 and instead of embedding just (x,ny) it embeds all 4 dims (x, y, w, h)
                query_sine_embed = gen_sineembed_for_position(
                    reference_points_input[:, :, 0, :]
                )  # nq, bs, 256*2
            else:
                query_sine_embed = gen_sineembed_for_position(
                    reference_points
                )  # nq, bs, 256*2
                reference_points_input = None

            # conditional query
            # project the reference_point positional embeddings (num_queries, b, hidden_dim)
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256

            assert raw_query_pos.shape[-1] == query_sine_embed.shape[-1] // 2

            # None by default so query_pos = raw_query_pos
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            # skipped by default
            if not self.deformable_decoder:
                query_sine_embed = query_sine_embed[
                    ..., : self.d_model
                ] * self.query_pos_sine_scale(output)

            # skipped by default; modulated HW attentions
            if not self.deformable_decoder and self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid()  # nq, bs, 2
                query_sine_embed[..., self.d_model // 2 :] *= (
                    refHW_cond[..., 0] / reference_points[..., 2]
                ).unsqueeze(-1)
                query_sine_embed[..., : self.d_model // 2] *= (
                    refHW_cond[..., 1] / reference_points[..., 3]
                ).unsqueeze(-1)

            dropflag = False

            # skipped by default (no layer dropping); random drop some layers if needed
            if self.dec_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.dec_layer_dropout_prob[layer_id]:
                    dropflag = True

            if not dropflag:
                # call the deformable transformer decoder layer which performs
                # self-attention, deformable cross-attention, and a two-layer ffn
                # output shape (num_queries, b, hidden_dim)
                output = layer(
                    tgt=output,
                    tgt_query_pos=query_pos,
                    tgt_query_sine_embed=query_sine_embed,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    tgt_reference_points=reference_points_input,  # refpoints were sigmoided
                    memory=memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    memory_level_start_index=level_start_index,
                    memory_spatial_shapes=spatial_shapes,
                    memory_pos=pos,
                    self_attn_mask=tgt_mask,
                    cross_attn_mask=memory_mask,
                )

            # iter update
            if self.bbox_embed is not None:
                # convert reference points to logits (num_queries, b, 4)
                reference_before_sigmoid = inverse_sigmoid(reference_points)

                # predict the bbox offset corrections for the decoder lyaer
                # (num_queries, b, hidden_dim) -> (num_queries, b, 4) where 4 = (cx, cy, w, h)
                delta_unsig = self.bbox_embed[layer_id](output)

                # add the embedded decoder layer boxes w/ the reference point logits
                # then apply sigmoid to bound [0, 1]
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                # skipped by default; select # ref points
                if (
                    self.dec_layer_number is not None
                    and layer_id != self.num_layers - 1
                ):
                    nq_now = new_reference_points.shape[0]
                    select_number = self.dec_layer_number[layer_id + 1]
                    if nq_now != select_number:
                        class_unselected = self.class_embed[layer_id](
                            output
                        )  # nq, bs, 91
                        topk_proposals = torch.topk(
                            class_unselected.max(-1)[0], select_number, dim=0
                        )[
                            1
                        ]  # new_nq, bs
                        new_reference_points = torch.gather(
                            new_reference_points,
                            0,
                            topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
                        )  # unsigmoid

                if self.rm_detach and "dec" in self.rm_detach:  # skipped by default
                    reference_points = new_reference_points
                else:  # implemented in DAB-DETR and DINO
                    # detach new reference points so that they do not affect the current layers
                    # backpropagation (Fig 6. (a))
                    # remaining layers will use these new detached `reference_points`
                    # even though they're not appended below
                    reference_points = new_reference_points.detach()

                # NOTE this is where the look forward twice module is implemented;
                # Look forward twice (lft) is designed to influence the current layer's (layer i)
                # parameters by losses of both layer i and layer i+1 (the next decoder layer) so
                # it can "look forward"
                if self.use_detached_boxes_dec_out:  # DAB-DETR uses this
                    ref_points.append(reference_points)
                else:
                    # Look forward twice (DINO DETR uses this);
                    # new_reference_points contains the
                    # intial reference points + the predicted decoder_layer bbox_embeds;
                    # this is what gets passed outside the model and into the loss TODO this isn't quite right
                    ref_points.append(new_reference_points)

            # store the raw decoder_layer outputs (before bbox_embed) for every
            # decoder layer and apply LayerNorm
            intermediate.append(self.norm(output))

            # skipped by default
            if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
                if nq_now != select_number:
                    output = torch.gather(
                        output,
                        0,
                        topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model),
                    )  # unsigmoid

        # return a list of the raw decoder layer outputs and a list of the refined reference points
        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
        ]


def _get_clones(module: nn.Module, N: int, layer_share: bool = False):
    """Creates a list of copies a `module` or layer `N` times; if
    layer_share=True, then the layers

    Args:
        module: nn.Module to create copies of
        N: number of module copies to make
        layer_share: whether to use the same module N times (this shares the same parameters)
                     or to make deep copies of the layer; default is to make deep copies
    """
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_deformable_transformer(
    num_feature_levels: int,
    query_dim: int,
    num_patterns: int,
    two_stage_args: dict[str, any],
    transformer_args: dict[str, any],
):
    """Builds the deformable transformer module

    Args:
        num_feature_levels: number of multiscale feature maps extracted from the backbone
        query_dim: the dimension of the reference point queries;
                   typically 4 for (cx, cy, w, h) or 2 for (x, y)
        num_patterns:TODO
        two_stage_args: TODO
        transformer_args: TODO
    """

    # This is False by default and never called; TODO: still need to briefly understand the code
    transformer_args["decoder_layer_noise"]
    if transformer_args["decoder_layer_noise"]:
        dec_noise_params = transformer_args["dec_noise_params"]
        decoder_query_perturber = RandomBoxPerturber(
            x_noise_scale=dec_noise_params["dln_xy_noise"],
            y_noise_scale=dec_noise_params["dln_xy_noise"],
            w_noise_scale=dec_noise_params["dln_hw_noise"],
            h_noise_scale=dec_noise_params["dln_hw_noise"],
        )
    else:
        decoder_query_perturber = None

    # False = Look forward twice (which is what DINO implements)
    use_detached_boxes_dec_out = transformer_args["use_detached_boxes_dec_out"]

    ####### start here and try initailize deformable transformer ########

    return DeformableTransformer(
        d_model=transformer_args["hidden_dim"],
        num_heads=transformer_args["num_heads"],
        num_obj_queries=transformer_args["num_queries"],
        num_encoder_layers=transformer_args["num_encoder_layers"],
        num_unicoder_layers=transformer_args["num_unicoder_layers"],
        num_decoder_layers=transformer_args["num_decoder_layers"],
        dim_feedforward=transformer_args["feedforward_dim"],
        dropout=transformer_args["dropout"],
        activation=transformer_args["activation"],
        normalize_before=transformer_args["pre_norm"],
        return_intermediate_dec=transformer_args["return_intermediate_dec"],
        query_dim=query_dim,
        num_patterns=num_patterns,
        modulate_hw_attn=True,  # set to True but it's never actually used
        deformable_encoder=True,
        deformable_decoder=True,
        num_feature_levels=num_feature_levels,
        enc_n_points=transformer_args["enc_n_points"],
        dec_n_points=transformer_args["dec_n_points"],
        use_deformable_box_attn=transformer_args["use_deformable_box_attn"],
        box_attn_type=transformer_args["box_attn_type"],
        learnable_tgt_init=True,
        decoder_query_perturber=decoder_query_perturber,
        add_channel_attention=transformer_args["add_channel_attention"],
        add_pos_value=transformer_args["add_pos_value"],
        random_refpoints_xy=False,
        two_stage_type=two_stage_args["type"],
        two_stage_pat_embed=two_stage_args["pat_embed"],
        two_stage_add_query_num=two_stage_args["add_query_num"],
        two_stage_learn_wh=two_stage_args["learn_wh"],
        two_stage_keep_all_tokens=two_stage_args["keep_all_tokens"],
        dec_layer_number=transformer_args["dec_layer_number"],
        rm_dec_query_scale=transformer_args["rm_dec_query_scale"],
        rm_self_attn_layers=transformer_args["rm_self_attn_layers"],
        key_aware_type=None,
        layer_share_type=None,
        rm_detach=transformer_args["rm_detach"],
        decoder_sa_type=transformer_args["decoder_self_attn_type"],
        module_seq=transformer_args["decoder_module_seq"],
        # for denoising
        embed_init_tgt=transformer_args["embed_init_tgt"],
        use_detached_boxes_dec_out=use_detached_boxes_dec_out,
    )
