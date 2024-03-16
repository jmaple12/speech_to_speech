# pip install cn2an, pypinyin, jieba_fast, pyopenjtalk, g2p_en, pip3 install ffmpeg-python
# cmd:python,import nltk,  nltk.download('cmudict')
import winsound
import sys
import re

#猜想：前期慢是路径太多了，
#搞了半天是因为我的os_path不对，改一下os路径就好了、
# 1.更改os.path，程序结尾将os.path改回来。
# 2.text_segmentation_method.py修改cut1，增加按照中英文句号问号叹号结束的cut6(cut6没更新成功)
# 3. TTS.py:因为我的python版本是3.8, 所以from typing import Tuple, 将 tuple改为Tuple

original_syspath = sys.path
all_path = sys.path[0]
all_path = re.sub('/','\\\\', all_path)

#清理sys.path中相同路径
syspathset = set()
copy_syspath=[]
for syspath in sys.path:
    if syspath not in syspathset:
        syspathset.add(syspath)
        copy_syspath.append(syspath)
sys.path = copy_syspath
del syspathset,copy_syspath

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

#来自i18n.py #将i18n.py的程序整合过来
import json
import locale


'''
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
'''
import os, sys
now_dir = os.getcwd()
change_ospath = sys.path[1]+'\\'+'\\'.join(all_path.split('\\')[:-1])
os.chdir(change_ospath)
#os.path更改后all_path也要与os.path对齐
all_path = all_path.split('\\')[-1]

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
bert_path =all_path+"\pretrained_models\chinese-roberta-wwm-ext-large"
cnhubert_base_path=all_path+"\pretrained_models\chinese-hubert-base"
gpt_path = all_path+"\pretrained_models\s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
sovits_path = all_path+"\pretrained_models\s2G488k.pth"

for some_path in [gpt_path, sovits_path, bert_path, cnhubert_base_path]:
    some_path = '/'.join(some_path.split('\\'))

# import gradio as gr
from TTS_infer_pack.TTS import TTS, TTS_Config
from TTS_infer_pack.text_segmentation_method import get_method

from tools.i18n.i18n import I18nAuto
i18n = I18nAuto()

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

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

tts_config = TTS_Config(all_path+"/configs/tts_infer.yaml")
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
tts_pipline = TTS(tts_config)
gpt_path = tts_config.t2s_weights_path
sovits_path = tts_config.vits_weights_path

def inference(text, text_lang, 
              ref_audio_path, prompt_text, 
              prompt_lang, top_k, 
              top_p, temperature, 
              text_split_method, batch_size, 
              speed_factor, ref_text_free,
              split_bucket
              ):
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
        "return_fragment":False
    }
    
    for item in tts_pipline.run(inputs):
        yield item
        
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
              split_bucket=False, audio_save=False, audio_save_file='output.wav'):
    with torch.no_grad():
        gen = inference(text, text_lang, 
              ref_audio_path, prompt_text, 
              prompt_lang, top_k, 
              top_p, temperature, 
              text_split_method, batch_size, 
              speed_factor, ref_text_free,
              split_bucket)
        sampling_rate, audio_data = next(gen)

    if audio_save:
        sf.write('output.wav', audio_data, sampling_rate, format="wav")
    #wave存储二进制音频
    wav = BytesIO()
    sf.write(wav, audio_data, sampling_rate, format="wav")
    wav.seek(0)
    #将二进制音频输出出来
    winsound.PlaySound(wav.read(), winsound.SND_MEMORY)

#把os.path改回来
os.chdir(now_dir)
sys.path = original_syspath[1:]