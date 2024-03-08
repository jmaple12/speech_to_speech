# speech_to_speech
Combine several AI model to achieve speech to speech, which wil also reduce some performance.

Now I begin to show the file folder meaning and some model needed to download. 

## voice_record
I define a function named **listen** to achieve automatically record man's voice and stop when speak is over. This function is here[LargeModel/voice_record/voice_record_def.py]
```        
def listen(WAVE_OUTPUT_FILENAME, tag, delayTime=2, tendure=2, mindb = 500):
return(sign, tag)
```
　　where the **WAVE_OUTPUT_FILENAME** is the recording file path,  sign=1 means it record nothing, **mindb** means the minimum decibel it will record. If it receives some sound before, the function will be end if it dosen't receive sound after **'delayTime'** seconds, and the function only endure **'tendure'** seconds blank sound.   
  
　　In **'LargeModel\Combine\combine.ipynb'**, your sound record will be cut several section, when your sound pause exceed 'delayTime' seconds, the sound will be saved in **'LargeModel\Combine\test'**, and if your sound pause exceed **'tendure'** seconds, the sound record will be end.  

## Sound to Text

　　In this section I use the Speech Recognition Model Fast_Whisper,we need to download the model according to [https://github.com/SYSTRAN/faster-whisper], or from [https://hf-mirror.com/Systran/faster-whisper-large-v3]　and put it in **'LargeModel/Speech_to_Text/Fast_whisper
/large-v3'** folder, and we can download other size model, it decided by our computer performance. 

## Large language Model
And then we need to download large LLM, I use Google Gemma[https://github.com/google/gemma_pytorch], and I use it through ollama[https://github.com/ollama/ollama], in ollama weibsite[https://ollama.com/], download its application, and execute below code in cmd
```
ollama run gemma:7b
or
ollama run gemma:2b
```
and then we can download and use the gemma model. 

## Text_to_Voice/Speech_Synthesis
### Text_to_Voice
**pyttsx3** can direcitly use the system voice package, and it consumes minimal resources but its voice is bad. 

### Speech_Synthesis
　　On the other hand, we can also use Speech_Synthesis Model, i use GPT_Sovits Model[https://github.com/RVC-Boss/GPT-SoVITS],its new version for windows is here[https://www.123pan.com/s/5tIqVv-GVRcv.html], it gives a package to process from audio cleaning to model_train and model_inference. It contains a pretains model about Paimon's voice, therefore, i use the pretained model and i only focus on the inference section. In order to use its inference model, we need to place the Integration package under the **'LargeModel/Speech_Synthesis/GPT_Sovits'** and place the **LargeModel/Speech_Synthesis
/GPT_Sovits/inference_maple.py** into the Integration package folder which contains **'inference_webui.py'** , and edit the **all_path** according to yourself filepath in the line 14 of the **'inference_maple.py'**.     

　　Notice：GPT_Sovits has its own environment, we need to install some package into ourself environment like package: cn2an, pypinyin, jieba_fast, pyopenjtalk, g2p_en, pip3 install ffmpeg-python and so on.

## Final 

open **'LargeModel/Combine/combine.ipynb'**, download some python package, modify variables in the third bloack 
```model_size, download_root, model_path, text_text_model
```
and variables in fourth block 
  ```
ref_wav_path, prompt_text, prompt_language, text, text_language, sovits_path, gpt_path
```
according to your condition.

## Summarize
　　I main make a framework which places several AI model to achieve speech to speech, and i focus on the localization model. Because this issue need to run at least 2 ai model at the same time, it requires large GPU video memory, and it will be worse than the AI model of speech conversation. My GPU only has 4G video memory, so i only run the framework in a low level, and the fast_whisper can't translate my voice well and the gpt_sovits read a sentense per 4 seconds. if you have high computer congifure, you can try use larger fast_whisper model and Gemma or other outstanding AI model. So far, the gemma and fast_whisper is the best miniaturization open_source model in its field.  
　　of course, we can also place api of internet ai model in this framework, it may achieve much better performance. I find Ernie Bot app has speech conversation function, it is good.    
  　I hope an ai loudspeaker box which can communicate with human fluently and intelligently as it is a man appear in 5 years, i really need them.


    
