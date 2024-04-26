需要将os.system设置在本目录，也就是stream_zipformer的父级目录。
stream_zipformer中的除了几个py文件外都是模型包sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar中的数据

sherpa_onnx 可进行中英文的实时转义。安装参考
https://zhuanlan.zhihu.com/p/664492030， 
https://k2-fsa.github.io/sherpa/onnx/python/install.html#method-1-from-pre-compiled-wheels
下载的CPU版本运行，可以GPU但是我没有安装好。

python版本实时语音转换：
https://k2-fsa.github.io/sherpa/onnx/python/real-time-speech-recongition-from-a-microphone.html

预训练模型在这里下载：https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
和
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html