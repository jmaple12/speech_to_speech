import os
import torch
import warnings
# warnings.filterwarnings("ignore")
import whisper

name='small'
model_path = 'E:\LargeModel\Speech_to_Text\Whisper\Model'
model= whisper.load_model(name, download_root=model_path)#device='cuda:0'
def whisper_output(model, file_path, initial_prompt = None):

    #initial_prompt:"以下是普通话的句子。"
    result = model.transcribe(audio= file_path,initial_prompt=initial_prompt)#是否显示时间戳
    return(result["text"])