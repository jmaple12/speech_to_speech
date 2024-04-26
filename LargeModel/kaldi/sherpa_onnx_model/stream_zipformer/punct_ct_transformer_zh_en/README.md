# Introduction

This model is converted from
https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary

The commands to generate the onnx model are given below:

```
pip install funasr modelscope
pip install kaldi-native-fbank torchaudio onnx onnxruntime

mkdir -p /tmp/models
cd /tmp/models

git clone https://www.modelscope.cn/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch.git
cd punc_ct-transformer_zh-cn-common-vocab272727-pytorch
git lfs pull --include model.pt

cd /tmp
git clone https://github.com/alibaba-damo-academy/FunASR
cd FunASR/runtime/python/onnxruntime

cat >export-onnx.py <<EOF

from funasr_onnx import CT_Transformer
model_dir = "/tmp/punc_ct-transformer_zh-cn-common-vocab272727-pytorch" # model = CT_Transformer(model_dir, quantize=True) model = CT_Transformer(model_dir)
EOF

chmod +x export-onnx.py

./export-onnx.py
```

You will find the exported
`model.onnx` file inside
`/tmp/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch`.

Now you can use ./add-model-metadata.py in this repo to add metadata to the generated
`model.onnx`.

Note that we don't provide `model.int8.onnx` because
```
punc_ct-transformer_zh-cn-common-vocab272727-pytorch fangjun$ ls -lh *.onnx
-rw-r--r--  1 fangjun  staff   279M Apr 12 11:08 model.onnx
-rw-r--r--  1 fangjun  staff   270M Apr 12 11:09 model_quant.onnx
```

`model.int8.onnx` does not have advantages in the file size for this model.
