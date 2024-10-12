import torch
from torch import nn
from typing import Any
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.misc import fill_holes_in_mask_scores

class ImageEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.no_mem_embed = sam_model.no_mem_embed #[1,1,256]
        self.image_encoder = sam_model.image_encoder
        self.num_feature_levels = sam_model.num_feature_levels
        self.prepare_backbone_features = sam_model. _prepare_backbone_features

    @torch.no_grad()
    def forward(self, image: torch.Tensor) ->tuple[torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor,torch.Tensor]:
        backbone_out = self.image_encoder(image) # {"vision_features","vision_pos_enc","backbone_fpn"}
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
       
        vision_pos_enc = backbone_out["vision_pos_enc"] # 有3个tensor
        backbone_fpn = backbone_out["backbone_fpn"]     # 有3个tensor
        pix_feat = backbone_out["vision_features"] # 有1个tensor

        expanded_backbone_out = {
            "backbone_fpn": backbone_fpn,
            "vision_pos_enc": vision_pos_enc,
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(1, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            expanded_backbone_out["vision_pos_enc"][i] = pos.expand(1, -1, -1, -1)
        
        (_,current_vision_feats,current_vision_pos_embeds,_) = self.prepare_backbone_features(expanded_backbone_out)

        current_vision_feat = current_vision_feats[-1] + self.no_mem_embed
        current_vision_feat2 = current_vision_feat.reshape(64,64,1,256).permute(2, 3, 0, 1) # [1,256,64,64]
        
        # flatten HWxNxC -> NxCxHxW
        high_res_features_0 = current_vision_feats[0].reshape(256,256, 1, 32).permute(2, 3, 0, 1) # [1, 32, 256, 256]
        high_res_features_1 = current_vision_feats[1].reshape(128,128, 1, 64).permute(2, 3, 0, 1) # [1, 64, 128, 128]

        # pix_feat              [1, 256, 64, 64]
        # high_res_features_0   [1, 32, 256, 256]
        # high_res_features_1   [1, 64, 128, 128]
        # current_vision_feat   [1, 256, 64, 64]
        # current_vision_pos_embed2 [4096, 1, 256]
        return pix_feat,high_res_features_0,high_res_features_1,current_vision_feat2,current_vision_pos_embeds[-1]

class MemAttention(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.no_mem_embed = sam_model.no_mem_embed
        self.memory_attention = sam_model.memory_attention

    # @torch.no_grad()
    def forward(
        self,
        current_vision_feat: torch.Tensor,      #[1, 256, 64, 64], 当前帧的视觉特征
        current_vision_pos_embed: torch.Tensor, #[4096, 1, 256], 当前帧的位置特征
        memory_1:torch.Tensor,                  # [num_obj_ptr,256]->[num_obj_ptr,4,64]->[4*num_obj_ptr,1,64]
        memory_2:torch.Tensor,                  # [n,64,64,64]->[n,64,4096]->[4096n,1,64] 
        memory_pos_1:torch.Tensor,              # [y*4096,1,64]
        memory_pos_2:torch.Tensor               # [num_obj_ptr,256]->[num_obj_ptr,4,64]->[4*num_obj_ptr,1,64]
    ) -> tuple[Any]:
        # num_obj_ptr_tokens =  memory_1.shape[0]*4
        current_vision_feat = current_vision_feat.permute(2,3,0,1).reshape(4096,1,256)
        current_vision_feat = current_vision_feat - self.no_mem_embed

        memory_2 = memory_2.view(-1, 64, 64*64).permute(0,2,1)
        memory_2 = memory_2.reshape(-1,1,64)

        memory_1 = memory_1.reshape(-1,1,4,64)
        memory_1 = memory_1.permute(0, 2, 1, 3).flatten(0, 1)

        # memory_pos_2 = memory_pos_2.reshape(-1,1,4,64)
        # memory_pos_2 = memory_pos_2.permute(0, 2, 1, 3).flatten(0, 1)

        self.memory_attention.allocate_rope_attention_weight(
            curr = current_vision_feat,
            curr_pos = current_vision_pos_embed
        )
        pix_feat_with_mem = self.memory_attention(
            curr = current_vision_feat,
            memory_1 = memory_2,
            memory_2 = memory_1,
            curr_pos = current_vision_pos_embed,
            memory_pos_1 = memory_pos_1,
            memory_pos_2 = memory_pos_2,
        )
        image_embed = pix_feat_with_mem.permute(1, 2, 0).view(1, 256, 64, 64) # [1,256,64,64]
        return image_embed #[1,256,64,64]

class MemEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.maskmem_tpos_enc = sam_model.maskmem_tpos_enc
        self.feat_sizes = [(256, 256), (128, 128), (64, 64)]
    @torch.no_grad()
    def forward(
        self,
        mask_for_mem: torch.Tensor,  # [1,1,1024,1024]
        pix_feat: torch.Tensor,      # [1,256,64,64]
    )-> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=pix_feat,
            feat_sizes=self.feat_sizes,
            pred_masks_high_res=mask_for_mem,
            is_mask_from_pts=True,
        )
        print(maskmem_features.shape)
        # maskmem_features = maskmem_features.view(1, 64, 64*64).permute(2, 0, 1)
        maskmem_pos_enc = maskmem_pos_enc.view(1, 64, 64*64).permute(2, 0, 1)

        return maskmem_features,maskmem_pos_enc,self.maskmem_tpos_enc

class ImageDecoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.sigmoid_scale_for_mem_enc = sam_model.sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sam_model.sigmoid_bias_for_mem_enc
    @torch.no_grad()
    def forward(
        self,
        point_coords: torch.Tensor, # [num_labels,num_points,2]
        point_labels: torch.Tensor, # [num_labels,num_points]
        frame_size: torch.Tensor,   # [2]
        image_embed: torch.Tensor,  # [1,256,64,64]
        high_res_feats_0: torch.Tensor, # [1, 32, 256, 256]
        high_res_feats_1: torch.Tensor, # [1, 64, 128, 128]
    ):
        point_inputs = {"point_coords":point_coords,"point_labels":point_labels}
        high_res_feats = [high_res_feats_0, high_res_feats_1]

        sam_outputs = self.model._forward_sam_heads(
            backbone_features=image_embed,
            point_inputs=point_inputs,
            mask_inputs=None,
            high_res_features=high_res_feats,
            multimask_output=True
        )
        (
            _,
            _,
            _,
            low_res_masks, # [1,1,256,256]
            high_res_masks, # [1,1,1024,1024]
            obj_ptr,  # [1,256]
            _,
        ) = sam_outputs
        # 处理高分辨率mask
        mask_for_mem = torch.sigmoid(high_res_masks)
        mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        # 填洞
        low_res_masks = fill_holes_in_mask_scores(low_res_masks, 8)
        # 还原到原图大小
        pred_mask = torch.nn.functional.interpolate(
            low_res_masks,
            size=(frame_size[0], frame_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        return obj_ptr,mask_for_mem,pred_mask