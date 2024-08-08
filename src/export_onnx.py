import onnx
import torch
from typing import Any
from torch import nn
from sam2.modeling.sam import mask_decoder
from sam2.modeling.sam import prompt_encoder
from sam2.modeling.sam import transformer
from sam2.modeling import memory_encoder
from sam2.modeling import memory_attention

class OnnxMemAttention(nn.Module):
    def __init__(self,memory_attention:memory_attention.MemoryAttention) -> None:
        super().__init__()
        self.memory_attention = memory_attention
    
    def forward(
        self,
        num_obj_ptr: torch.Tensor,              #缓存的obj_ptr数量,官方是缓存了16帧
        current_vision_feat: torch.Tensor,      #[4096, 1, 256], 当前帧的视觉特征
        current_vision_pos_embed: torch.Tensor, #[4096, 1, 256], 当前帧的位置特征
        memory:torch.Tensor,                    #[y*4096,1,64], 最近y帧的记忆编码特性
        memory_pos_embed:torch.Tensor,          #[y*4096,1,64], 最近y帧的位置编码特性
    ) -> tuple[Any]:
        
        num_obj_ptr_tokens =  int(num_obj_ptr[0][0]*4)

        print("num_obj_ptr_tokens = ",num_obj_ptr_tokens)
        print("current_vision_feat  = ",current_vision_feat.shape)
        print("current_vision_pos_embed  = ",current_vision_pos_embed.shape)
        print("memory  = ",memory.shape)
        print("memory_pos_embed  = ",memory_pos_embed.shape)

        pix_feat_with_mem = self.memory_attention(
            curr = current_vision_feat,
            curr_pos = current_vision_pos_embed,
            memory = memory,
            memory_pos = memory_pos_embed,
            num_obj_ptr_tokens= num_obj_ptr_tokens,
        )
        # reshape the output (HW)xBxC => BxCxHxW
        image_embed = pix_feat_with_mem.permute(1, 2, 0).view(1, 256, 64, 64) # [1,256,64,64]
        return image_embed #[1,256,64,64]


model_config_file = "sam2_configs/sam2_hiera_t.yaml"
model_checkpoints_file = "checkpoints/tiny/memory_attention.pt"

self_attention = transformer.RoPEAttention(
        rope_theta=10000.0,
        feat_sizes=[32, 32],
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1)
cross_attention = transformer.RoPEAttention(
        rope_theta= 10000.0,
        feat_sizes= [32, 32],
        rope_k_repeat= True,
        embedding_dim= 256,
        num_heads= 1,
        downsample_rate= 1,
        dropout= 0.1,
        kv_in_dim= 64)
layer = memory_attention.MemoryAttentionLayer(
        activation = "relu",
        dim_feedforward = 2048,
        dropout = 0.1,
        pos_enc_at_attn = False,
        self_attention = self_attention,
        d_model = 256,
        pos_enc_at_cross_attn_keys = True,
        pos_enc_at_cross_attn_queries = False,
        cross_attention = cross_attention)

# 实例化模型
model_mem_atten = memory_attention.MemoryAttention(
    d_model = 256,
    pos_enc_at_input = True,
    layer = layer,
    num_layers= 4
)

# 加载模型参数,确保模型处于评估模式
model_mem_atten.load_state_dict(torch.load(model_checkpoints_file))
model_mem_atten.eval()
onnx_model = OnnxMemAttention(model_mem_atten)

num_obj_ptr = torch.Tensor([[16]])
current_vision_feat = torch.ones(4096,1,256)       #[4096, 1, 256],当前帧的视觉特征
current_vision_pos_embed = torch.ones(4096,1,256)  #[4096, 1, 256],当前帧的位置特征
memory = torch.ones(7*4096+64,1,64)                    #[y*4096,1,64], 最近y帧的记忆编码特性
memory_pos_embed = torch.ones(7*4096+64,1,64)          #[y*4096,1,64], 最近y帧的位置编码特性

r = onnx_model(
    num_obj_ptr = num_obj_ptr,
    current_vision_feat = current_vision_feat,
    current_vision_pos_embed = current_vision_pos_embed,
    memory = memory,
    memory_pos_embed = memory_pos_embed
)

print(r)

input_name = [  "num_obj_ptr",
                "current_vision_feat",
                "current_vision_pos_embed", 
                "memory",
                "memory_pos_embed"]

onnx_filename = "checkpoints/tiny/mem_tiny.onnx"

args=(
    num_obj_ptr,
    current_vision_feat,
    current_vision_pos_embed,
    memory,
    memory_pos_embed
)
dynamic_axes = {
    "num_obj_ptr":{0: "num"},
    "memory": {0: "buff"},
    "memory_pos_embed": {0: "buff"}
}

# version 1 with  error:
# return output_adapter.apply(model_func(*args, **kwargs), model=model)
# TypeError: OnnxMemAttention.forward() got an unexpected keyword argument 'export_params'
# torch.onnx.OnnxExporterError: Failed to export the model to ONNX. Generating SARIF report at 'report_dynamo_export.sarif'. SARIF is a standard format for the output of static analysis tools. SARIF logs can be loaded in VS Code SARIF viewer extension, or SARIF web viewer (https://microsoft.github.io/sarif-web-component/). Please report a bug on PyTorch Github: https://github.com/pytorch/pytorch/issues
torch.onnx.dynamo_export(
    onnx_model,
    (
        num_obj_ptr,
        current_vision_feat,
        current_vision_pos_embed,
        memory,
        memory_pos_embed
    ),
    onnx_filename,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names= input_name,
    output_names=["image_embed"],
    dynamic_axes = dynamic_axes
    ).save(onnx_filename)

# version 2 with RuntimeError: ScalarType ComplexFloat is an unexpected tensor scalar type
# torch.onnx.export(
#     onnx_model,
#     (
#         num_obj_ptr,
#         current_vision_feat,
#         current_vision_pos_embed,
#         memory,
#         memory_pos_embed
#     ),
#     onnx_filename,
#     export_params=True,
#     opset_version=17,
#     do_constant_folding=True,
#     input_names= input_name,
#     output_names=["image_embed"],
#     dynamic_axes = dynamic_axes
#     ).save(onnx_filename)