import os
os.environ['HF_ENDPOINT']='hf-mirror.com'
# cmd:set HF_ENDPOINT=https://hf-mirror.com
from faster_whisper import WhisperModel
import warnings
import sys
import glob
import re

def fast_whisper_opt(model, input_audio, initial_prompt=None, transcribe_params = {}):

    '''
    input_audio:输入音频位置
    model_size:模型尺寸：'small', 'large-v3'
    '''  
    # model_size='large-v3'
    # download_root=r'E:\LargeModel\Speech_to_Text\fast_whisper'
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16", download_root=download_root, local_files_only=False)  
    # input_audio = r"D:\Desktop\test.mp3"

    #jupyter死机的原因是D:\Anaconda\Library 中有一个libiomp5md.dl，与其他地方的同一个文件冲突
    segments,_ = model.transcribe(
        input_audio, 
        initial_prompt=initial_prompt,
        word_timestamps=False,
        **transcribe_params
        )
    segments = ','.join([segment.text for segment in segments])
    return(segments)
# print(fast_whisper_opt(r'D:\Desktop\test.wav'))

#path[0]为当前目录
original_path = sys.path
# sys.path.append('\\'.join(sys.path[0].split('\\')[:-2]))
sys.path.append(os.path.join(sys.path[0], os.pardir, os.pardir))#LargeModel
from voice_record.voice_record_def import listen#记录输入的语音

#对话前删除临时录音文件
def delete_files(folder_path):
    file_list = glob.glob(folder_path + '/*')
    for file_path in file_list:
        if os.path.isfile(file_path):
            if re.search('test\d+\..{3}$', file_path):
                os.remove(file_path)

def fast_whisper_realtime(audio_file_path, model, delayTime=0.8, tendure=3, mindb=2000, transcribe_params={}, initial_prompt=None):
    output_word=''
    result=''
    # 调用函数删除文件夹中的文件
    delete_files(audio_file_path)
    audio_file_path += '\\test'
    audio_num=1
    sign=0
    tag=0
    while(sign!=1): 
        out_audio = audio_file_path+str(audio_num)+'.wav'
        sign, tag = listen(out_audio, tag, delayTime=delayTime, tendure=tendure, mindb = mindb)
        #jupyter死机的原因是D:\Anaconda\Library 中有一个libiomp5md.dl，与其他地方的同一个文件冲突
        if sign !=1:
            result = fast_whisper_opt(model, out_audio, initial_prompt=initial_prompt, transcribe_params=transcribe_params)
            audio_num +=1
            print(result, end='')
            output_word += result
    return(result, output_word)

sys.path = original_path