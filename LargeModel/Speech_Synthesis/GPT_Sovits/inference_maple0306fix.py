# pip install cn2an, pypinyin, jieba_fast, pyopenjtalk, g2p_en, pip3 install ffmpeg-python
# cmd:python,import nltk,  nltk.download('cmudict')
import winsound
import sys

# 更改的地方：
#1.源程序几个path的路径都写错了，不应该有GPT_Sovits/
#2.all_path主要用于定位bert_path和cnhubert_base_path
# 3.此外本程序里的load_language_list和I18nAuto里面的路径也要改
# 4.517行detach的[0,0]位置改动
# 5.i18n.py的内容我直接放在本文件里面了
# 6.增加了禁用进度条的程序tqdm_replacement函数
#7. 增加了get_streaming_tts_wav流式传输函数
#8. 重写cut1函数，cut5中的punds添加英文:，更改了punds中重复的;
#9. 删除i18n相关代码_i18n是用来读取系统语言，然后将用户输入的母语指令转为程序内可识别的指令。都是中国人，我不需要这一段
#10. 新增cut6"按中英文句末标识符切"(中英文的句号问号叹号，超过一个.会变成空格)，修改get_tts_wav切句后的'/n'合并的语句。

#只需要将本变量改为整合包根目录的文件路径
all_path = "E:\LargeModel\Speech_Synthesis\GPT_Sovits\GPT-SoVITS-beta\GPT-SoVITS-beta0306fix"
#this inference_maple.py file in my computer is locate at "E:\LargeModel\Speech_Synthesis\GPT_Sovits\GPT-SoVITS-beta\GPT-SoVITS-beta0306fix\GPT_SoVITS\inference_maple.py"

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

#get_tts_wav 中的prompt_language和text_language注释掉

#-----------------------------------------------------------------------------------------------------------#
#下面来自inference_weui.py----从头到cut5函数为止
'''
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
'''
import os, re, logging
import LangSegment
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import pdb
import torch

# if os.path.exists("./gweight.txt"):
#     with open("./gweight.txt", 'r', encoding="utf-8") as file:
#         gweight_data = file.read()
#         gpt_path = os.environ.get(
#             "gpt_path", gweight_data)
# else:
#     gpt_path = os.environ.get(
#         "gpt_path", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")

# if os.path.exists("./sweight.txt"):
#     with open("./sweight.txt", 'r', encoding="utf-8") as file:
#         sweight_data = file.read()
#         sovits_path = os.environ.get("sovits_path", sweight_data)
# else:
#     sovits_path = os.environ.get("sovits_path", "GPT_SoVITS/pretrained_models/s2G488k.pth")
# gpt_path = os.environ.get(
#     "gpt_path", "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
# )
# sovits_path = os.environ.get("sovits_path", "pretrained_models/s2G488k.pth")
cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)
infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and not torch.backends.mps.is_available()
# import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
from feature_extractor import cnhubert

bert_path =all_path+"\GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large"
cnhubert_base_path=all_path+"\GPT_SoVITS\pretrained_models\chinese-hubert-base"

for some_path in [bert_path, cnhubert_base_path]:#[gpt_path, sovits_path, bert_path, cnhubert_base_path]
    some_path = '/'.join(some_path.split('\\'))

from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from my_utils import load_audio

cnhubert.cnhubert_base_path = cnhubert_base_path

#清理内存
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()


os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)


def change_sovits_weights(sovits_path):
    global vq_model, hps
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    with open("./sweight.txt", "w", encoding="utf-8") as f:
        f.write(sovits_path)


# change_sovits_weights(sovits_path)


def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./gweight.txt", "w", encoding="utf-8") as f: f.write(gpt_path)


# change_gpt_weights(gpt_path)


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


dict_language = {
    "中文": "all_zh",#全部按中文识别
    "英文": "en",#全部按英文识别#######不变
    "日文": "all_ja",#全部按日文识别
    "中英混合": "zh",#按中英混合识别####不变
    "日英混合": "ja",#按日英混合识别####不变
    "多语种混合": "auto",#多语种启动切分识别语种
}


def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text

dtype=torch.float16 if is_half == True else torch.float32
def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def get_phones_and_bert(text,language):
    if language in {"en","all_zh","all_ja"}:
        language = language.replace("all_","")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 因无法区别中日文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        phones, word2ph, norm_text = clean_text_inf(formattext, language)
        if language == "zh":
            bert = get_bert_feature(norm_text, word2ph).to(device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja","auto"}:
        textlist=[]
        langlist=[]
        LangSegment.setfilters(["zh","ja","en","ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "ko":
                    langlist.append("zh")
                    textlist.append(tmp["text"])
                else:
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    return phones,bert.to(dtype),norm_text


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result



#来自0302版本的inference_stream.py: from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()

def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p, temperature, ref_free, stream):

    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    t0 = ttime()
    # prompt_language = dict_language[prompt_language]
    # text_language = dict_language[text_language]
    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
        print("实际输入的参考文本:", prompt_text)
    text = text.strip("\n")
    if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text
    
    print("实际输入的目标文本:", text)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
            raise OSError("参考音频在3~10秒范围外，请更换！")
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half == True:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(
            1, 2
        )  # .float()
        codes = vq_model.extract_latent(ssl_content)
   
        prompt_semantic = codes[0, 0]
    t1 = ttime()

    if (how_to_cut == "凑四句一切"):
        text = cut1(text)
    elif (how_to_cut == "凑50字一切"):
        text = cut2(text)
    elif (how_to_cut == "按中文句号。切"):
        text = cut3(text)
    elif (how_to_cut == "按英文句号.切"):
        text = cut4(text)
    elif (how_to_cut == "按标点符号切"):
        text = cut5(text)
    elif (how_to_cut == "按中英文句末标识符切"):
        text = cut6(text)
    text = re.sub(r'\n+','\n', text)

    print("实际输入的目标文本(切句后):", text)
    texts = text.split("\n")
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    if not ref_free:
        phones1,bert1,norm_text1=get_phones_and_bert(prompt_text, prompt_language)

    for text in texts:
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        if (text[-1] not in splits): text += "。" if text_language != "en" else "."
        print("实际输入的目标文本(每句):", text)
        phones2,bert2,norm_text2=get_phones_and_bert(text, text_language)
        print("前端处理后的文本(每句):", norm_text2)
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1+phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = prompt_semantic.unsqueeze(0).to(device)
        t2 = ttime()
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                None if ref_free else prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=hz * max_sec,
            )
        t3 = ttime()
        # print(pred_semantic.shape,idx)
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(
            0
        )  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(hps, ref_wav_path)  # .to(device)
        if is_half == True:
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = (
            vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
            )
                .detach()[0, 0]
                .cpu()
                .numpy()
        )  ###试试重建不带上prompt部分
        max_audio=np.abs(audio).max()#简单防止16bit爆音
        if max_audio>1:audio/=max_audio
        if stream:
            # 流式模式下每句返回一次
            yield (np.concatenate([audio, zero_wav], 0) * 32768).astype(np.int16).tobytes()
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
    # print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))

    if not stream:
        yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
            np.int16
    )


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    opt=[]
    while(len(inps)>4):
        opt.append(''.join(inps[:4]))
        inps = inps[4:]
    if len(inps)>0:
        opt.append(''.join(inps))
    return('\n'.join(opt))

    # split_idx = list(range(0, len(inps), 4))
    ## split_idx[-1] = None #此句话错误
    # split_idx.append(None)
    # if len(split_idx) > 1:
    #     opts = []
    #     for idx in range(len(split_idx) - 1):
    #         opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    # else:
    #     opts = [inp]
    # return "\n".join(opts)

def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip("。").split("。")])


def cut4(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip(".").split(".")])


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    # if not re.search(r'[^\w\s]', inp[-1]):
    # inp += '。'
    inp = inp.strip("\n")
    punds = r'[,.;?!、，。？！；：:…]'
    items = re.split(f'({punds})', inp)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    # 在句子不存在符号或句尾无符号的时候保证文本完整
    if len(items)%2 == 1:
        mergeitems.append(items[-1])
    opt = "\n".join(mergeitems)
    return opt

#根据中英文的句号问号叹号分割，跳过省略号
def cut6(inp):
    inp = inp.strip("\n")
    inp = re.sub(r'\.{2,}',' ', inp)
    return(re.sub(r'[。?？！!\.]','\n', inp))


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

def get_streaming_tts_wav(
    ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p, temperature, ref_free
):    
    chunks = get_tts_wav(
        ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p, temperature, ref_free, True
    )
    yield wave_header_chunk()
    for chunk in chunks:
        yield chunk

def handle(ref_wav_path, prompt_text, prompt_language, text, text_language, sovits_path, gpt_path, start=False, top_k=5, top_p=1, temperature=1, how_to_cut="凑四句一切", ref_free = False, stream=True):
    '''
    sovits_path:sovits_weight_path
    gpt_path:gpt_weight_path
    start：第一次使用模型时先启动系统，text默认是'。'为True则禁止用语音。
    stream：是否流式输出
    '''
    change_sovits_weights(sovits_path)
    change_gpt_weights(gpt_path)

    if not stream:

        if start:
            if text =='':
                text='。'
        with torch.no_grad():
            gen = get_tts_wav(
                ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p, temperature, ref_free, stream
            )
            sampling_rate, audio_data = next(gen)
        if not start:
            #wave存储二进制音频
            wav = BytesIO()
            sf.write(wav, audio_data, sampling_rate, format="wav")
            wav.seek(0)
            #将二进制音频输出出来
            winsound.PlaySound(wav.read(), winsound.SND_MEMORY)
    
    else:
        p = pyaudio.PyAudio()
        #来自inference_maple.py的wave_header_chunk
        stream = p.open(format = p.get_format_from_width(2),
                        channels = 1,
                        rate = 32000,
                        frames_per_buffer=4096,
                        output = True)
        for chunk in get_streaming_tts_wav(
            ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p, temperature, ref_free):
            stream.write(chunk)
        stream.stop_stream()#暂停
        stream.close()#关闭
        p.terminate()
        
#恢复记录和进度条---调用段使用
# sys.stdout = original_stdout
# import tqdm
tqdm.tqdm = tqdm_copy
