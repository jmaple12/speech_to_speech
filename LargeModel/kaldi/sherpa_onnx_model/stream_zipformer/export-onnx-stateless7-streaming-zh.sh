#!/usr/bin/env bash

. path.sh

export CUDA_VISIBLE_DEVICES=

./pruned_transducer_stateless7_streaming/export-onnx-zh.py \
  --lang-dir ./k2fsa-zipformer-chinese-english-mixed/data/lang_char_bpe \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --exp-dir ./k2fsa-zipformer-chinese-english-mixed/exp/ \
  --decode-chunk-len 32 \
  --num-encoder-layers "2,4,3,2,4" \
  --feedforward-dims "1024,1024,1536,1536,1024" \
  --nhead "8,8,8,8,8" \
  --encoder-dims "384,384,384,384,384" \
  --attention-dims "192,192,192,192,192" \
  --encoder-unmasked-dims "256,256,256,256,256" \
  --zipformer-downsampling-factors "1,2,4,8,2" \
  --cnn-module-kernels "31,31,31,31,31" \
  --decoder-dim 512 \
  --joiner-dim 512
