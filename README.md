# speech_to_speech
Combine several AI model to achieve speech to speech, which also reduce some performance.

Now I begin to show the file folder meaning and some model needed to download. 

## voice_record
I define a function named listen to achieve automatically record man's voice and stop when speak is over.  
···        
def listen(WAVE_OUTPUT_FILENAME, tag, delayTime=2, tendure=2, mindb = 500):
return(sign, tag)
···            
where the WAVE_OUTPUT_FILENAME is the recording file path,  sign=1 means it record nothing. If it receives some sound before, the function will be end if it dosen't receive sound after 'delayTime' seconds, and the function only endure 'tendure' seconds blank sound.   
In 'LargeModel\Combine\combine.ipynb', your sound record will be cut several section, when your sound pause exceed 'delayTime' seconds, the sound will be saved in 'LargeModel\Combine\test', and if your sound pause exceed 'tendure' seconds, the sound record will be end.  

