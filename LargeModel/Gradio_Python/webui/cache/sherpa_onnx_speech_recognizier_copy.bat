chcp 65001
cd /d E:/LargeModel/kaldi/sherpa_onnx_model
python stream_zipformer/speech-recognition-from-microphone-with-endpoint-detection-maple.py --tokens=stream_zipformer/tokens.txt --encoder=stream_zipformer/encoder-epoch-99-avg-1.onnx --decoder=stream_zipformer/decoder-epoch-99-avg-1.onnx --joiner=stream_zipformer/joiner-epoch-99-avg-1.onnx