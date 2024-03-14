# Speech_to_speech

　　Combine several AI model to achieve speech to speech, which wil also reduce some performance. I only share my way in windows environment, pytorch and cuda is needed.     

　　Now I begin to show the file folder meaning and some model needed to download. 

## voice_record

　　I define a function named **listen** to achieve automatically record man's voice and stop when speak is over. This function is here[voice_record_def.py](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/voice_record/voice_record_def.py)
```        
def listen(WAVE_OUTPUT_FILENAME, tag, delayTime=2, tendure=2, mindb = 500):
return(sign, tag)
```
　　where the **WAVE_OUTPUT_FILENAME** is the recording file path,  sign=1 means it record nothing, **mindb** means the minimum decibel it will record. If it has received some sound before, the function will be end if it dosen't receive sound after **delayTime** seconds, and the function only endure **tendure** seconds blank sound, the sound record will stop if the blank sound is longer.   
  
　　In [combine.ipynb](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Combine/combine.ipynb), your sound record will be cut several section, when your sound pause exceed **delayTime** seconds, the sound will be temporarily saved in [test](https://github.com/jmaple12/speech_to_speech/tree/main/LargeModel/Combine/test) file, and if your sound pause exceed **tendure** seconds, the sound record will be end until the next round conversation is start.  

## Sound to Text/ASR
### faster_whisper  
　　In this section I use the Speech Recognition Model Fast_Whisper, we need to download the model according to [fast-whisper](https://github.com/SYSTRAN/faster-whisper), or from [huggingface-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) or its mirror site [mirror-large-v3](https://hf-mirror.com/Systran/faster-whisper-large-v3)　and put it in [large-v3](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Speech_to_Text/Fast_whisper/large-v3) folder, and we can also download other size model, it decided by our computer performance.   

### sherpa
　　Because of poor video memory, I also try to use other tools. sherpa is a model based on kaldi, I　use its onnx version. Because I fail to install its GPU version, therefore, I use its CPU version. Its performance is better than my expectation. Firstly, it can translate the voice as soon as I say it, though it doesn't translate correctly as better as fast_whisper, I think its performance is OK for medium requirement. And some of their models supports both Chinese and English.   

  install sherpa_onnx windows can be found on[sherpa_onnx](https://k2-fsa.github.io/sherpa/onnx/install/windows.html#bit-windows-x64), and its python package install can be found on [sherpa-onnx python](https://k2-fsa.github.io/sherpa/onnx/python/install.html#method-1-from-pre-compiled-wheels). Its real-time-speech-recongition issue can be found on[real-time-speech-recongition](https://k2-fsa.github.io/sherpa/onnx/python/real-time-speech-recongition-from-a-microphone.html), and its pretrained model can be download on[pretrained model](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models). 

  In [LargeModel/kalid/sherpa_onnx/microphone_endpoint.py](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/kalid/sherpa_onnx/microphone_endpoint.py), I modify the [python-api-examples/speech-recognition-from-microphone-with-endpoint-detection.py](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/speech-recognition-from-microphone-with-endpoint-detection.py) so that I can directly use its "main" function in python.   

### whisper
　　Besides, [fast_whisper.ipynb](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Speech_to_Text/Fast_whisper/fast_whisper.ipynb) can automatically create caption file(.srt) of an audio, it may reduce work for somebody.    

## Large language Model

　　And then we need to download large LLM, I use [Google Gemma](https://github.com/google/gemma_pytorch), and I use it through [ollama](https://github.com/ollama/ollama), in [ollama weibsite](https://ollama.com/), download its application, and execute below code in cmd
```
ollama run gemma:7b
or
ollama run gemma:2b
```
and then we can use the gemma model. 

## Text_to_Voice/Speech_Synthesis

### Text_to_Voice

　　**pyttsx3** can directly use the windows system voice package, and it consumes minimal resources but its voice is bad. 

### Speech_Synthesis

　　On the other hand, we can also use Speech_Synthesis Model to create our own sound, I use GPT_Sovits Model[GPT_Sovits Model](https://github.com/RVC-Boss/GPT-SoVITS),its new version for windows is here[windows Integration package0306](https://www.123pan.com/s/5tIqVv-GVRcv.html), it gives a package to process from audio cleaning to audio_model_train and model_inference. It contains a pretained model about Paimon's voice, I think it is enough, therefore, I use its pretained model and then I only focus on its inference section.   
  
  In order to use its inference model, we need to place the Integration package under the [GPT_Sovits](https://github.com/jmaple12/speech_to_speech/tree/main/LargeModel/Speech_Synthesis/GPT_Sovits) and place the [inference_maple.py](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Speech_Synthesis/GPT_Sovits/inference_maple.py) into the Integration package folder which contains **inference_webui.py** , and edit the variables **all_path** according to yourself filepath in the line 14 of the [inference_maple.py](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Speech_Synthesis/GPT_Sovits/inference_maple.py).      

　　Besides, [gpt_sovits_api.ipynb](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Speech_Synthesis/GPT_Sovits/gpt_sovits_api.ipynb) gives two ways to inference, and second way can directly use its inference section without downloading python module to yourself environment, it only need to run the api.py in cmd before excute the code, and api may cause more time delay.    

　　Notice：GPT_Sovits has its own environment, if we use the first way in  [gpt_sovits_api.ipynb](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Speech_Synthesis/GPT_Sovits/gpt_sovits_api.ipynb)  or [combine.ipynb](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Combine/combine.ipynb) we need to install some package into ourself environment like package: cn2an, pypinyin, jieba_fast, pyopenjtalk, g2p_en, ffmpeg-python and so on.

## Final 

　　open [combine.ipynb](https://github.com/jmaple12/speech_to_speech/blob/main/LargeModel/Combine/combine.ipynb), download some python package, modify variables in the third bloack 
```
model_size, download_root, model_path, text_text_model
```
and variables in fourth block 
  ```
ref_wav_path, prompt_text, prompt_language, text, text_language, sovits_path, gpt_path
```
according to your condition.

## Summarize

　　I mainly make a framework which suture several AI model to achieve speech to speech, and I focus on the localization model. Because this issue need to run at least 2 AI model at the same time, it requires large GPU video memory, and it will be worse than the speech conversation AI model. My GPU only has 4G video memory, so I only run the framework in a low level, fast_whisper can't translate my voice well and the gpt_sovits slowly deal with a sentense per 4 seconds. if you have higher computer congifure, you can try to use larger fast_whisper and Gemma  model or other outstanding AI model.  
  
　　of course, we can also place api of internet AI model in this framework, it may achieve much better performance may cause more time delay. I find Ernie Bot app has speech conversation function, and it is good.    
  
  　I hope an AI loudspeaker box which can communicate with human fluently and intelligently as it is a man appear in 5 years, I really need them.


    
