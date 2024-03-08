import os
os.environ['HF_ENDPOINT']='hf-mirror.com'
# cmd:set HF_ENDPOINT=https://hf-mirror.com
import os
import torch
from faster_whisper import WhisperModel
import warnings
# warnings.filterwarnings("ignore")
# #清理内存
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# if hasattr(torch.cuda, 'empty_cache'):
#     torch.cuda.empty_cache()


def fast_whisper_opt(model, input_audio, initial_prompt=None):

    '''
    input_audio:输入音频位置
    model_size:模型尺寸：'small', 'large-v3'
    '''  
    # model_size='large-v3'
    # download_root=r'E:\LargeModel\Speech_to_Text\fast_whisper'
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16", download_root=download_root, local_files_only=False)  
    # input_audio = r"D:\Desktop\test.mp3"

    #jupyter死机的原因是D:\Anaconda\Library 中有一个libiomp5md.dl，与其他地方的同一个文件冲突

    segments,_ = model.transcribe(input_audio, initial_prompt=initial_prompt, word_timestamps=False)
    segments = ','.join([segment.text for segment in segments])
    return(segments)
# print(fast_whisper_opt(r'D:\Desktop\test.wav'))