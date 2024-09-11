# SAM2Export
将sam2导出为onnx文件。

## 下载pt模型文件

```sh
cd checkpoints
./download_ckpts.sh
mkdir base+ large small tiny
```

## run

python export_onnx.py

## 参考

SAM2: https://github.com/facebookresearch/segment-anything-2.git

fix SAM2: https://github.com/axinc-ai/segment-anything-2.git
