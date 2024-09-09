import torch
import onnx
import argparse
from onnxsim import simplify
from src.Module import ImageEncoder
from src.Module import MemAttention
from src.Module import MemEncoder
from src.Module import ImageDecoder
from sam2.build_sam import build_sam2
from hydra import compose, initialize

def export_image_encoder(model,onnx_path):
    input_img = torch.randn(1, 3,1024, 1024).cpu()
    out = model(input_img)
    output_names = ["pix_feat","high_res_feat0","high_res_feat1","vision_feats","vision_pos_embed"]
    torch.onnx.export(
        model,
        input_img,
        onnx_path+"image_encoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=output_names,
    )
    # # 简化模型, tmd将我的输出数量都简化掉一个，sb
    # original_model = onnx.load(onnx_path+"image_encoder.onnx")
    # simplified_model, check = simplify(original_model)
    # onnx.save(simplified_model, onnx_path+"image_encoder.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"image_encoder.onnx")
    onnx.checker.check_model(onnx_model)
    print("image_encoder.onnx model is valid!")
    
def export_mem_attention(model,onnx_path):
    num_obj_ptr = torch.tensor([16],dtype=torch.int32)
    current_vision_feat = torch.randn(1,256,64,64)       #[1, 256, 64, 64],当前帧的视觉特征
    current_vision_pos_embed = torch.randn(4096,1,256)  #[4096, 1, 256],当前帧的位置特征
    memory = torch.randn(7*4096+64,1,64)                    #[y*4096,1,64], 最近y帧的记忆编码特性
    memory_pos_embed = torch.randn(7*4096+64,1,64)          #[y*4096,1,64], 最近y帧的位置编码特性
    out = model(
            num_obj_ptr = num_obj_ptr,
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            memory = memory,
            memory_pos_embed = memory_pos_embed
        )

    input_name = ["num_obj_ptr",
                "current_vision_feat",
                "current_vision_pos_embed",
                "memory",
                "memory_pos_embed"]
    dynamic_axes = {
        "num_obj_ptr":{0: "num"},
        "memory": {0: "buff"},
        "memory_pos_embed": {0: "buff"}
    }
    torch.onnx.export(
        model,
        (num_obj_ptr,current_vision_feat,current_vision_pos_embed,memory,memory_pos_embed),
        onnx_path+"mem_attrntion.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["image_embed"],
        dynamic_axes = dynamic_axes
    )
     # 简化模型,
    original_model = onnx.load(onnx_path+"mem_attrntion.onnx")
    simplified_model, check = simplify(original_model)
    onnx.save(simplified_model, onnx_path+"mem_attrntion.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"mem_attrntion.onnx")
    onnx.checker.check_model(onnx_model)
    print("mem_attrntion.onnx model is valid!")

def export_image_decoder(model,onnx_path):
    point_coords = torch.randn(1,2,2).cpu()
    point_labels = torch.randn(1,2).cpu()
    frame_size = torch.tensor([1024,1024],dtype=torch.int32)
    image_embed = torch.randn(1,256,64,64).cpu()
    high_res_feats_0 = torch.randn(1,32,256,256).cpu()
    high_res_feats_1 = torch.randn(1,64,128,128).cpu()

    out = model(
        point_coords = point_coords,
        point_labels = point_labels,
        frame_size = frame_size,
        image_embed = image_embed,
        high_res_feats_0 = high_res_feats_0,
        high_res_feats_1 = high_res_feats_1
    )
    input_name = ["point_coords","point_labels","frame_size","image_embed","high_res_feats_0","high_res_feats_1"]
    output_name = ["obj_ptr","mask_for_mem","pred_mask"]
    dynamic_axes = {
        "point_coords":{0: "num_labels",1:"num_points"},
        "point_labels": {0: "num_labels",1:"num_points"}
    }
    torch.onnx.export(
        model,
        (point_coords,point_labels,frame_size,image_embed,high_res_feats_0,high_res_feats_1),
        onnx_path+"image_decoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=output_name,
        dynamic_axes = dynamic_axes
    )
    # 简化模型,
    original_model = onnx.load(onnx_path+"image_decoder.onnx")
    simplified_model, check = simplify(original_model)
    onnx.save(simplified_model, onnx_path+"image_decoder.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"image_decoder.onnx")
    onnx.checker.check_model(onnx_model)
    print("image_decoder.onnx model is valid!")

def export_memory_encoder(model,onnx_path):
    mask_for_mem = torch.randn(1,1,1024,1024) 
    pix_feat = torch.randn(1,256,64,64) 

    out = model(mask_for_mem = mask_for_mem,pix_feat = pix_feat)

    input_names = ["mask_for_mem","pix_feat"]
    output_names = ["maskmem_features","maskmem_pos_enc","temporal_code"]
    torch.onnx.export(
        model,
        (mask_for_mem,pix_feat),
        onnx_path+"memory_encoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_names,
        output_names= output_names
    )
    # 简化模型,
    # original_model = onnx.load(onnx_path+"memory_encoder.onnx")
    # simplified_model, check = simplify(original_model)
    # onnx.save(simplified_model, onnx_path+"memory_encoder.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_encoder.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_encoder.onnx model is valid!")


#****************************************************************************
model_type = ["tiny","small","large","base+"][3]
onnx_output_path = "checkpoints/{}/".format(model_type)
model_config_file = "sam2_hiera_{}.yaml".format(model_type)
model_checkpoints_file = "checkpoints/sam2_hiera_{}.pt".format(model_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出SAM2为onnx文件")
    parser.add_argument("--outdir",type=str,default=onnx_output_path,required=False,help="path")
    parser.add_argument("--config",type=str,default=model_config_file,required=False,help="*.yaml")
    parser.add_argument("--checkpoint",type=str,default=model_checkpoints_file,required=False,help="*.pt")
    args = parser.parse_args()
    sam2_model = build_sam2(args.config, args.checkpoint, device="cpu")

    # image_encoder = ImageEncoder(sam2_model).cpu()
    # export_image_encoder(image_encoder,args.outdir)

    # image_decoder = ImageDecoder(sam2_model).cpu()
    # export_image_decoder(image_decoder,args.outdir)


    mem_attention = MemAttention(sam2_model).cpu()
    export_mem_attention(mem_attention,args.outdir)

    mem_encoder   = MemEncoder(sam2_model).cpu()
    export_memory_encoder(mem_encoder,args.outdir)