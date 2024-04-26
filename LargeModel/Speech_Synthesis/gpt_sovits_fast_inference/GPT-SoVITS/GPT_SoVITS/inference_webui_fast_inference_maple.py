# pip install cn2an, pypinyin, jieba_fast, pyopenjtalk, g2p_en, pip3 install ffmpeg-python
# cmd:python,import nltk,  nltk.download('cmudict')
import winsound
import sys
import re
import os
now_dir = os.getcwd()
original_path = sys.path
#猜想：前期慢是路径太多了，
#搞了半天是因为我的os_path不对，改一下os路径就好了、
# 0.在windows的环境变量-系统变量中添加 变量PYTHONPATH:GPT_SOVITS父级路径
    #本程序路径"E:\LargeModel\Speech_Synthesis\gpt_sovits_inference\GPT_SoVITS"
    #则将E:\LargeModel\Speech_Synthesis\gpt_sovits_inference"添加到环境变量
    #或在调用本程序前将GPT_SOVITS父级路径添加到sys.path中
# 1.更改将os.path更为GPT_SOVITS的父级路径，程序结尾将os.path改回来。
# 2.text_segmentation_method.py修改cut1，增加按照中英文句号问号叹号结束的cut6
# 3.TTS.py:因为我的python版本是3.8, 所以from typing import Tuple, 将 tuple改为Tuple
# 4.加入handle函数。
# 5.将sovit_path和gpt_path 作为handle的自变量引入，好像会慢一些。
# 6.增加了禁用进度条的程序tqdm_replacement函数
# 7. GPT_SoVITS\TTS_infer_pack\TextPreprocessor.py中的pre_seg_text函数在文本最前面加了句号（行75），现在改为在文本最后面加句号。更改切句后的将多个'/n'合并为一个的语句（原本行83-84），在“text = text.strip("\n")”后面加上一句“if not text: text = '\n'”防止text变为空字符串；104行左右在函数pre_seg_text的末尾增加防止texts为空列表的代码：if not texts:texts.append("。") if lang != "en" else texts.append(".")（这个主要是chrome_api 输出空音频的时候会结束，实际返回空列表是对的）
#8. api_v2_maple.py。根据api_v2.py修改，运行脚本的时候增加-s,-g的模型路径参数来传入模型路径，在\tts中增加gpt_weights和sovits_weights参数来初始化tts的模型参数(\tts这个尽量不使用，因为每读取一小段文本都要重新初始化模型，会让api变慢，尽量在\set_gpt_weights和\set_sovits_weights中永久更改config中的自定义模型路径。)

#将os.path锁定为当前文件所在路径
curr_path= [content for content in sys.path if re.match('.+:.*gpt_sovits_inference$', content)]
#找到包含GPT_SOVITS的文件路径
for path_temp in curr_path:
    if 'GPT_SOVITS' in path_temp:
        break
curr_path = path_temp
os.chdir(curr_path)
sys.path.append(curr_path+'\\GPT_SOVITS')

#清理sys.path中相同路径
syspathset = set()
copy_syspath=[]
for syspath in sys.path:
    if syspath not in syspathset:
        syspathset.add(syspath)
        copy_syspath.append(syspath)
sys.path = copy_syspath
del syspathset

# 将标准输出流重定向到空设备（/dev/null）---在调用层执行
# original_stdout =  sys.stdout
# sys.stdout = open('log_maple.txt', 'w')

#禁用进度条------进度条终于没有了！！！！！！！！！！！！！！！！！
import tqdm
def tqdm_replacement(iterable_object,*args,**kwargs):
    return iterable_object
tqdm_copy = tqdm.tqdm # store it if you want to use it later
tqdm.tqdm = tqdm_replacement

#来自0302的inference_stream.py
import wave
import pyaudio

#来自api.py中的扩展包---最后音频的流式输出用到了它
from io import BytesIO
import soundfile as sf

'''
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
'''
import random
import logging
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import pdb
import torch

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and not torch.backends.mps.is_available()

gpt_path = os.environ.get("gpt_path", None)
sovits_path = os.environ.get("sovits_path", None)
cnhubert_base_path = os.environ.get("cnhubert_base_path", None)
bert_path = os.environ.get("bert_path", None)

#默认模型路径
bert_path ="GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large"
cnhubert_base_path="GPT_SoVITS\pretrained_models\chinese-hubert-base"
gpt_path = "GPT_SoVITS\pretrained_models\s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
sovits_path = "GPT_SoVITS\pretrained_models\s2G488k.pth"

for some_path in [gpt_path, sovits_path, bert_path, cnhubert_base_path]:
    some_path = '/'.join(some_path.split('\\'))

# import gradio as gr
from TTS_infer_pack.TTS import TTS, TTS_Config
from TTS_infer_pack.text_segmentation_method import get_method

from tools.i18n.i18n import I18nAuto
i18n = I18nAuto()

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

if torch.cuda.is_available():
    device = "cuda"
# elif torch.backends.mps.is_available():
#     device = "mps"
else:
    device = "cpu"
    
dict_language = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
}

cut_method = {
    i18n("不切"):"cut0",
    i18n("凑四句一切"): "cut1",
    i18n("凑50字一切"): "cut2",
    i18n("按中文句号。切"): "cut3",
    i18n("按英文句号.切"): "cut4",
    i18n("按标点符号切"): "cut5",
    i18n("按中英文句末标识符切"):"cut6"
}

tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
tts_config.device = device
tts_config.is_half = is_half
if gpt_path is not None:
    tts_config.t2s_weights_path = gpt_path
if sovits_path is not None:
    tts_config.vits_weights_path = sovits_path
if cnhubert_base_path is not None:
    tts_config.cnhuhbert_base_path = cnhubert_base_path
if bert_path is not None:
    tts_config.bert_base_path = bert_path
    
print(tts_config)
tts_pipeline = TTS(tts_config)
gpt_path = tts_config.t2s_weights_path
sovits_path = tts_config.vits_weights_path


def inference(text, text_lang, 
              ref_audio_path, prompt_text, 
              prompt_lang, top_k, 
              top_p, temperature, 
              text_split_method, batch_size, 
              speed_factor, ref_text_free,
              split_bucket,
              return_fragment, fragment_interval,
              seed,
              ):
    actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)
    inputs={
        "text": text,
        "text_lang": dict_language[text_lang],
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text if not ref_text_free else "",
        "prompt_lang": dict_language[prompt_lang],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": cut_method[text_split_method],
        "batch_size":int(batch_size),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "return_fragment":return_fragment,
        "fragment_interval":fragment_interval,
        "seed":actual_seed,
    }
    for item in tts_pipeline.run(inputs):
        yield item, actual_seed
        
def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


pretrained_sovits_name = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
SoVITS_weight_root = "SoVITS_weights"
GPT_weight_root = "GPT_weights"
os.makedirs(SoVITS_weight_root, exist_ok=True)
os.makedirs(GPT_weight_root, exist_ok=True)

def handle(text, text_lang, 
              ref_audio_path, prompt_text, 
              prompt_lang, top_k=5, 
              top_p=1, temperature=1, 
              text_split_method='按标点符号切', batch_size=1, 
              speed_factor=1, ref_text_free=False,
              return_fragment=False,fragment_interval=0.3,seed=-1, 
              split_bucket=False, 
              audio_save=False, audio_save_file='output.wav', 
              mygpt_path=None, mysovits_path=None, play=True):
    '''
    text:输入待推理文本，text_lang:待推理文本的语言，ref_audio_path:参考音频路径
    prompt_text:参考音频的文本，prompt_lang:参考音频的文本的语言
    text_split_method:文本切分方式，batch_size:推理时同时进行的线程数，speed_factor:语速
    ref_text_free:是否使用参考音频的文本，audio_save:是否保存音频
    return_fragment:是否流式输出,
    fragment_interval：流式输出时音频分段间隔（秒）,seed：随机种子
    audio_save_file:音频保存的位置，mygpt_path:gpt_weight的路径
    mysovits_path:sovits_weight模型的路径，play：是否播放推理出的音频
    '''
    os.chdir(curr_path)
    if mygpt_path is None:
        mygpt_path = gpt_path
    if mysovits_path is None:
        mysovits_path = sovits_path
    
    if not os.path.samefile(tts_pipeline.configs.vits_weights_path, mysovits_path):
        tts_pipeline.init_vits_weights(mysovits_path)
    if not os.path.samefile(tts_pipeline.configs.t2s_weights_path, mygpt_path):
        tts_pipeline.init_t2s_weights(mygpt_path) 
    
    with torch.no_grad():
        gen = inference(text, text_lang, 
            ref_audio_path, prompt_text, 
            prompt_lang, top_k, 
            top_p, temperature, 
            text_split_method, batch_size, 
            speed_factor, ref_text_free,
            split_bucket,
            return_fragment, fragment_interval,
            seed,)
    if not return_fragment:
        item, seed = next(gen)
        os.chdir(now_dir)
        sampling_rate, audio_data = item
        if audio_save:
            sf.write(audio_save_file, audio_data, sampling_rate, format="wav")
        if play:
            #wave存储二进制音频
            wav = BytesIO()
            sf.write(wav, audio_data, sampling_rate, format="wav")
            wav.seek(0)
            #将二进制音频输出出来
            winsound.PlaySound(wav.read(), winsound.SND_MEMORY)
    
    else:
        # 播放音频
        p = pyaudio.PyAudio()
        #来自inference_maple.py的wave_header_chunk
        stream = p.open(format = p.get_format_from_width(2),
                        channels = 1,
                        rate = 32000,
                        frames_per_buffer=4096,
                        output = True)
        if audio_save:
            wf = wave.open(audio_save_file, 'wb')  
            wf.setnchannels(1)  
            wf.setsampwidth(2)  
            wf.setframerate(32000) 
            wf.writeframes(b'') 
        for item, seed in gen:
            data = item[1]
            import time
            print(f"Writing data at {time.time()}: {len(data)} bytes")
            if play:
                stream.write(data)
            if audio_save:
                wf.writeframes(data)
        stream.stop_stream()#暂停
        stream.close()#关闭
        p.terminate()

#把os.path改回来
os.chdir(now_dir)
sys.path = original_path