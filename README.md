# speech_to_speech
Combine several AI model to achieve speech to speech, which also reduce some performance.

Now I begin to show the file folder meaning and some model needed to download. 

## voice_record
I define a function named listen to achieve automatically record man's voice and stop when speak is over.   

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
　　On the other hand, we can also use Speech_Synthesis Model, i use GPT_Sovits Model[https://github.com/RVC-Boss/GPT-SoVITS], it gives a package to process from audio cleaning to model train and model inference. It contains a pretains model about Paimon's voice, therefore, i use the pretained model and i only focus on the inference section. In order to use its inference model, we need to place the **LargeModel/Speech_Synthesis
/GPT_Sovits/inference_maple.py** into the folder which contains **'inference_webui.py'** in its Integration package, and 
