[English Introduction Please Click Here](https://github.com/jmaple12/speech_to_speech/blob/main/README_ENGLISH.md)

## 更新20240427

  现在基于ollama(LLM模型平台), kalidi-sherpa-onnx(ASR模型), gpt_sovits(TTS模型)以及Gradio，我得到了一个简易的本地语音聊天的网页。

  需要下载仓库中的
  [LargeModel/Gradio_Python](https://github.com/jmaple12/speech_to_speech/tree/main/LargeModel/Gradio_Python)、
  [LargeModel/kaldi/sherpa_onnx_model](https://github.com/jmaple12/speech_to_speech/tree/main/LargeModel/kaldi/sherpa_onnx_model)
  以及Github上的[GPT_SOVITS的fast_inference分支](https://github.com/RVC-Boss/GPT-SoVITS/tree/fast_inference_)，在其主目录加入本仓库GPT_SOVITS中的[api_v2开头的三个文件](https://github.com/jmaple12/speech_to_speech/tree/main/LargeModel/Speech_Synthesis/gpt_sovits_fast_inference/GPT-SoVITS)，下载Ollama并安装里面的模型。

  打开[Gradio Webui](https://github.com/jmaple12/speech_to_speech/tree/main/LargeModel/Gradio_Python/webui)，双击里面的 run.bat 运行网页，初次运行需要在“chat"栏目下的两个文本框内写入[TTS-API](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/kaldi/sherpa_onnx_model/sherpa_onnx_speech_recognizier.bat)以及[ASR-API](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Speech_Synthesis/gpt_sovits_fast_inference/GPT-SoVITS/api_v2_maple.bat)的文件路径。

# 本地语音对话

　　我在本项目里缝合了多个AI模型，来实现语音对话的功能，当然硬将多个不同功能的模型糅合在一起会浪费一些计算机性能。
    这里我仅仅分享自己在windows系统中如何搭建多个AI模型，然后缝合的代码在[languagemodel](https://github.com/jmaple12/speech_to_speech/tree/main/LargeModel)里面。为了使用AI模型，首先得下载GPU版本的torch以及对应的cudnn。 
    
　　下面我将要介绍[languagemodel](https://github.com/jmaple12/speech_to_speech/tree/main/LargeModel)里各个子文件夹的意义以及需要下载的东西，我主要展示自己的缝合代码。

## 录音

　　我定义了“Listen”函数来实现自动录音，当人声停止的时候，它能自动停止录音。这个函数在这里[voice_record_def.py](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/voice_record/voice_record_def.py)。
```        
def listen(WAVE_OUTPUT_FILENAME, tag, delayTime=2, tendure=2, mindb = 500):
return(sign, tag)
```
　　其中**WAVE_OUTPUT_FILENAME**是设定的存放录音文件的路径，“Listen”里的“sign”变量用来记录该次录音是否记录到了声音，当“sign=1”时表明本次录音什么声音都没有录到。**mindb**表示能录到的最小分贝值。如果之前录到了声音，且之后连续**delayTime** 秒没有接收到新的声音，录音会停止， 如果全程持续**tendure**秒没有录到声音，录音也会停止，并且此时“sign=1”来表示整个录音过程都没有录到声音。 

　　在[LargeModel/Combine/combine.ipynb](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Combine/combine.ipynb)的录音板块中，一轮录音会执行多次“listen”函数。我们输入语音的时候，每次稍长时间的停顿（超过**delayTime**秒），“Listen”就会记录一次录音结果，将此波次的录音音频存储到指定文件夹[test](https://github.com/jmaple12/speech_to_speech/tree/main/LargeModel/Combine/test)，并执行下一波次的“listen”函数，直到“listen”函数接收到一波时长超过**tendure** 秒的空白录音，到此一轮录音结束。

　　在[LargeModel/Combine/combine.ipynb](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Combine/combine.ipynb)里面，每有记录一波录音，专门的ASR模型会自动将其转为文本，并与之前转换的文本连接起来，在整轮录音及转换文字完成后送到文本生成模型中去。
  
## 声音转文本/ASR
### faster_whisper  
　　这一部分我首先介绍语音识别模型Faster_whisper，它是对Whisper进行C语言的重新编译而得到，与Whisper相比占用更低，速度更快。首先我们需要在[fast-whisper](https://github.com/SYSTRAN/faster-whisper)查看模型介绍，在[huggingface-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3)或镜像站[mirror-large-v3](https://hf-mirror.com/Systran/faster-whisper-large-v3)下载模型，large-v3版本的faster_whisper在int8版本下大约需要3.5G显存，float16版本下大约需要4G多显存，如果显存低于4G，可以去下载它的small或者medium版本。需要注意large-v3的训练集里面有中文，其他低版本的训练集里面没有中文。此外， [fast_whisper.ipynb](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Speech_to_Text/Fast_whisper/fast_whisper.ipynb) 给出了为音频文件生成(srt)格式字幕的办法。

   faster_whisper转换地比较准确，甚至可以转换出标点符号，不过录音的时候如果不说话，它可能会转换处奇怪的话，或者出现错误。而且转换略有延时。

### sherpa
　　因为我的电脑性能较差，除了深度学习大模型，我也尝试了对设备要求较低的模型，比如这个onnx版本的sherpa。它是基于新一代kaldi（小米集团语音首席科学家Daniel Povey开发的）的机器学习语音识别模型。优点就是识别特别快，基本上能做到实时语音转换，并且它的有些预训练模型同时支持中英文转换。其次就是它是轻量级模型，在CPU上基本就能实现实时语音转换，对机器要求低，甚至可以在安卓机上运行。但是缺点是与fast_whisper相比，它还不够准确，只能说对要求不高且设备性能有限的人来说够用，而且它无法转换出来标点符号。它的CPU版本安装比较容易，教程也比较多，但是GPU版本我至今没有安装成功，GPU版本的性能应该比CPU好很多。  

　　安装Windows版本的sherpa_onnx可以在[sherpa_onnx](https://k2-fsa.github.io/sherpa/onnx/install/windows.html#bit-windows-x64)找到，根据[sherpa-onnx python](https://k2-fsa.github.io/sherpa/onnx/python/install.html#method-1-from-pre-compiled-wheels)可以安装它的python扩展包，它的实时语音转换教程可以在[real-time-speech-recongition](https://k2-fsa.github.io/sherpa/onnx/python/real-time-speech-recongition-from-a-microphone.html)找到，除此之外还有根据音频文件转录文本的教程。它的预训练模型可以在[pretrained model github](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)或[pretrained model ks-fsa](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html)下载。“新一代Kaldi”是他们的公众号，b站也有他们的账号，还有演示。  

　　为了在python中直接调用语音转文字的功能，在[LargeModel/kalid/sherpa_onnx/microphone_endpoint.py](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/kalid/sherpa_onnx/microphone_endpoint.py)中，我修改了官方[python-api-examples/speech-recognition-from-microphone-with-endpoint-detection.py](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/speech-recognition-from-microphone-with-endpoint-detection.py)实时语音检测的部分代码。官方代码是用在cmd中的。在python中直接调用microphone_endpoint.py的main函数就可以进行录音并输出转换的文字了。

### whisper
　　whisper可以在[github:openai/whisper](https://github.com/openai/whisper)找到它的详情，[large-v3](https://hf-mirror.com/openai/whisper-large-v3)是它在huggingface镜像站的large-v3模型仓库。用法跟faster_whisper类似，github里面有它的介绍。

## 大语言模型

　　前面步骤做好后就需要下载文本生成的大语言模型了。我使用的是 [Google Gemma](https://github.com/google/gemma_pytorch)，它是新出的在同尺寸开源模型中表现最好的。我通过ollama使用它，ollama的官网是[ollama weibsite](https://ollama.com/)。在官网下载ollama软件后，在cmd执行下面命令
```
ollama run gemma:7b
or
ollama run gemma:2b
```
　　它会自动下载对应版本的gemma模型，调用方法可以参见仓库里面[Gemme_worse_the_cmd.ipynb](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Language_Model/Gemma/Gemme_worse_the_cmd.ipynb)程序，它支持流式生成文本。

## TTS/语音合成

### TTS(Text to Speech)

　　**pyttsx3** 是python的一个扩展包，他能直接调用Windows系统的语音包，它对设备要求低，但是它发出的声音很单一，也不好听。除此之外，还有名为**speech**的扩展包，它比**pyttsx3**更轻量简易，但是声音更糟糕。

### 语音合成

　　另一方面，我们可以使用语音合成模型来生成自定义的声音。我使用的是[GPT_Sovits Model](https://github.com/RVC-Boss/GPT-SoVITS)模型，这个模型现在有很多分支模型，在b站搜索名字GPT Sovits能看到很多。目前官方的最新版本是[windows Integration package0306fix](https://www.123pan.com/s/5tIqVv-GVRcv.html)。这是针对Windows用户的整合包，包含了前期音频降噪人声分离等操作所需工具，囊括了从音频清洗、模型训练到推理全过程，全程在网页操作，对新人比较友好。整合包里面已经包含了一个派蒙的预训练模型，我个人感觉这个模型的质量已经挺好了。因为这个整合包主要基于weiui操作，为了能在我们自己的python环境中使用，需要对包里面的inference_webui.py进行修改。

　　将[inference_maple0306fix.py](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Speech_Synthesis/GPT_Sovits/inference_maple0306fix.py)放在[GPT_Sovits](https://github.com/jmaple12/speech_to_speech/tree/main/LargeModel/Speech_Synthesis/GPT_Sovits)下面，使用时直接调用里面的handle函数即可。
  
　　此外，[gpt_sovits_api.ipynb](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Speech_Synthesis/GPT_Sovits/gpt_sovits_api.ipynb)给出了两种文本转语音的推理方式。第一种就是调用前面所说的handle函数，优点是参数都可以自定义，但是需要先在自己的python中配置inference_maple0306fix.py所需的扩展包。第二种只需要将[api.bat](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Speech_Synthesis/GPT_Sovits/api.bat)放入整合包根目录，双击打开即可使用，缺点就是模型推理所需要的模型需要在整合包的config.py中修改。   

　　注意：GPT_Sovits整合包里面内置python环境，如果我们使用[gpt_sovits_api.ipynb](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Speech_Synthesis/GPT_Sovits/gpt_sovits_api.ipynb)的第一种方法或者使用[combine.ipynb](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Combine/combine.ipynb)，我们需要安装python扩展包如cn2an, pypinyin, jieba_fast, pyopenjtalk, g2p_en, ffmpeg-python等。

## 结尾

　　打开[combine.ipynb](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Combine/combine.ipynb), 根据实际情况修改第三个代码块里的一些变量值
```
model_size, download_root, model_path, text_text_model
```
及第四个代码块的一些变量值
  ```
ref_wav_path, prompt_text, prompt_language, text, text_language, sovits_path, gpt_path
```

## 总结

　　我主要做了一个缝合各个大模型实现语音对话的大框架，同时我主要关注的是本地实现语音对话。由于这个项目需要至少同时运行两种不同的语言模型，因此它对GPU的要求比较高，如果有直接能实现语音对话的AI模型，这个项目就差太多了。我的电脑只有4G的显存，运行这个项目捉襟见肘，因此我基本上都是运行最小尺寸的模型，对于更好的设备，可以考虑使用更大尺寸的模型。目前不足的地方有ASR这一块语音识别的不够准或者不够快，然后gpt_sovits进行TTS的延时差不多有3秒，太慢了。

　　当然，未来也可以考虑将各个本地大模型使用线上模型的API代替，这样应该可以快很多，而且生成的文本和语音也会好很多。然后最近我发现文心一言手机APP可以进行语音对话，希望它能更好。

　　我希望未来5年内市场上能有一款足够智能的音响，它能像人一样流畅地智能地与人聊天，未来我肯定很需要它！

    
