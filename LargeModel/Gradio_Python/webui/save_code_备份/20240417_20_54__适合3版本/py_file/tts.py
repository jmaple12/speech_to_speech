import requests
import winsound
import os
import sys
import wave
import pyaudio
import gradio as gr
#先双击打开"E:\LargeModel\Speech_Synthesis\GPT_Sovits\GPT-SoVITS-beta\GPT-SoVITS-beta_fast_inference_0316\api_v2_maple.bat"
#首次加载很慢
#推理的web界面的怎么切，top_k， top_p,temperature参数在 api_v2_maple.py locate at [gpt_vosits/0316 fast_inference]

#model_path格式：{’gpt_path':, 'sovits_path':}

default_tts_api_path = 'E:\LargeModel\Speech_Synthesis\GPT_Sovits\GPT-SoVITS-beta\GPT-SoVITS-beta_fast_inference_0316'

default_model_path = {}
# model_path = {'gpt_path':'E:\LargeModel\Speech_Synthesis\GPT_SOVITS_0训练集\自己的模型\姬如千泷\训练集_denoise\yueer_denoise-e25.ckpt', 'sovits_path':'E:\LargeModel\Speech_Synthesis\GPT_SOVITS_0训练集\自己的模型\姬如千泷\训练集_denoise\yueer_denoise_e25_s575.pth'}
# model_path = {'gpt_path':'GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt', 'sovits_path':'GPT_SoVITS/pretrained_models/s2G488k.pth'}

default_params = {
    'ref_audio_path':"E:\LargeModel\Speech_Synthesis\GPT_SOVITS_0训练集\自己的模型\姬如千泷\训练集_denoise\这是一个传承了千年尊贵的名字。.wav",
    'prompt_text': '这是一个传承了千年尊贵的名字。',
    'prompt_lang':'zh',
    'text_lang':'zh',
    'seed':4242, 
    "text_split_method": "cut5",
    "batch_size": 8,
    "speed_factor":1.1,
    "streaming_mode": True,
}
audio_save_file = '.\cache\audio.wav'

def is_path(abs_path ,path):
    '''如果path为相对路径，则将其转为相对于abs_path的绝对路径；然后判断path是否存在'''
    if not os.path.isabs(path):
        path = os.path.join(abs_path, path)
    if os.path.exists(path):
        return(True)
    return(False)


class TTS():
    def __init__(self, model_path=default_model_path, params=default_params, play=True, audio_save_file='', tts_api_path=default_tts_api_path):
        self.model_path = model_path
        self.params = params
        self.play = play
        self.audio_save_file = audio_save_file
        self.tts_api_path = tts_api_path
        if self.model_path:
            if 'gpt_path' in self.model_path:
                if is_path(self.tts_api_path, self.model_path['gpt_path']):
                    requests.get('http://127.0.0.1:9880/set_gpt_weights', {'weights_path':self.model_path['gpt_path']})
                else:
                    print('gpt_path is not available')
                
            if 'sovits_path' in self.model_path:
                if is_path(self.tts_api_path, self.model_path['sovits_path']):
                    requests.get('http://127.0.0.1:9880/set_sovits_weights', {'weights_path':self.model_path['sovits_path']})
                else:
                    print('sovits_path is not available')

    def modify_model(self, model_path):
        #将新的model_path写入gpt_sovits 中的tts_infer.yaml文件中
        self.model_path = model_path
        if 'gpt_path' in self.model_path:
            if is_path(self.tts_api_path, self.model_path['gpt_path']):
                requests.get('http://127.0.0.1:9880/set_gpt_weights', {'weights_path':self.model_path['gpt_path']})
            else:
                print('gpt_path is not available')
            
        if 'sovits_path' in self.model_path:
            if is_path(self.tts_api_path, self.model_path['sovits_path']):
                requests.get('http://127.0.0.1:9880/set_sovits_weights', {'weights_path':self.model_path['sovits_path']})
            else:
                print('sovits_path is not available')
        
    def modify_params(self, params):
        self.params.update(params)
    
    def run(self, text):
        self.params.update({"text":text})
        url = 'http://127.0.0.1:9880/tts'
        if 'streaming_mode' in self.params:
            out_stream = self.params['streaming_mode']
        else:
            out_stream = False
        if not out_stream:
            result = requests.get(url, self.params).content
        if self.play and not out_stream:
            winsound.PlaySound(result, winsound.SND_MEMORY)
        if self.audio_save_file:
            wf = wave.open(self.audio_save_file, 'wb')  
            wf.setnchannels(1)  # 声道数，单声道为 1，立体声为 2   
            wf.setsampwidth(2)  # 样本宽度（字节），对于 16 位音频为 2
            wf.setframerate(32000) # 采样率
            wf.writeframes(b'') # 写入文件头
            if not out_stream:
                wf.writeframes(result)

        #流式传输的播放和存储
        if out_stream:
            p = pyaudio.PyAudio()
            #来自inference_maple.py的wave_header_chunk
            if self.play:
                stream = p.open(format = p.get_format_from_width(2),
                                channels = 1,
                                rate = 32000,
                                frames_per_buffer=4096,
                                output = True)
            for chunk in requests.get(url, self.params):
                if self.play:
                    stream.write(chunk)
                if self.audio_save_file:
                    wf.writeframes(chunk)
            if self.play:
                stream.stop_stream()#暂停
                stream.close()#关闭
                p.terminate()


def params_overall_json_update(json, gpt_path,sovits_path,ref_audio_path, prompt_text, batch_size,prompt_lang,text_lang,seed,text_split_method,speed_factor):
    model_path_name = ['gpt_path', 'sovits_path']
    params_name = ['ref_audio_path', 'prompt_text','batch_size','prompt_lang','text_lang','seed','text_split_method','speed_factor']
    for name in model_path_name:
        if eval(name):
            if 'model_path' not in json:
                json['model_path'] = {}
            json['model_path'].update({name:eval(name)})
    
    for name in params_name:
        if eval(name):
            if 'params' not in json:
                json['params'] = {}
            json['params'].update({name:eval(name)})
    json = gr.Json.update(value=json)
    return(json)
    
  
        



    