import requests
import winsound
import os,re
import json
import sys
import wave
import pyaudio
import gradio as gr
#先双击打开"E:\LargeModel\Speech_Synthesis\GPT_Sovits\GPT-SoVITS-beta\GPT-SoVITS-beta_fast_inference_0316\api_v2_maple.bat"
#首次加载很慢
#推理的web界面的怎么切，top_k， top_p,temperature参数在 api_v2_maple.py locate at [gpt_vosits/0316 fast_inference]

#model_path格式：{’gpt_path':, 'sovits_path':}

default_tts_api_path = 'E:\LargeModel\Speech_Synthesis\GPT_Sovits\GPT-SoVITS-beta\GPT-SoVITS-beta_fast_inference_0316'

default_model_path = {'gpt_path':'GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt', 'sovits_path':'GPT_SoVITS/pretrained_models/s2G488k.pth'}
# model_path = {'gpt_path':'E:\LargeModel\Speech_Synthesis\GPT_SOVITS_0训练集\自己的模型\姬如千泷\训练集_denoise\yueer_denoise-e25.ckpt', 'sovits_path':'E:\LargeModel\Speech_Synthesis\GPT_SOVITS_0训练集\自己的模型\姬如千泷\训练集_denoise\yueer_denoise_e25_s575.pth'}

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
audio_save_file = './cache/audio.wav'

splits = {'\n', ',','.','。','，','!','！', '?', '？',}

def is_path(abs_path ,path):
    '''如果path为相对路径，则将其转为相对于abs_path的绝对路径；然后判断path是否存在'''
    if not os.path.isabs(path):
        path = os.path.join(abs_path, path)
    if os.path.exists(path):
        return(True)
    return(False)


class TTS():
    def __init__(self, default_model_path =default_model_path, params=default_params, default_params=default_params, play=True, audio_save_file='', tts_api_path=default_tts_api_path, cache_path='./', splits=splits, endure_sentence=1,display=False):
        self.default_model_path = default_model_path
        self.params = params
        self.default_params = default_params
        self.play = play
        self.audio_save_file = audio_save_file
        self.tts_api_path = tts_api_path
        self.open_tts = True
        self.display=display

        self.cache_path = cache_path
        self.splits = splits
        self.endure_sentence=endure_sentence
    def modify_model(self, model_path={}):

        #将新的model_path写入gpt_sovits 中的tts_infer.yaml文件中
        path_name = ['gpt_path','sovits_path']
        model_tail_name = {'gpt_path':'ckpt' ,'sovits_path':'pth'}
        request_url = {'gpt_path':'http://127.0.0.1:9880/set_gpt_weights',
                        'sovits_path': 'http://127.0.0.1:9880/set_sovits_weights'
                        }
        #如果路径存在且非None，传入tts_api，更改config.tts_infer.yaml中的model_path
        #删除路径包含的可能的引号，model_path非空时用非空部分替换self.model_path,否者使用default_model_path
        self_model_path = self.default_model_path
        if self.open_tts:
            if model_path:
                for model_path_key, model_path_val in model_path.items():
                    if model_path_val:
                        self_model_path[model_path_key] = model_path_val.strip('\'').strip('"')
            for pth_name in path_name:
                if (is_path(self.tts_api_path, self_model_path[pth_name]))&(self_model_path[pth_name].split('.')[-1] ==model_tail_name[pth_name]):
                    print("TTS.py更新%s为:%s"%(pth_name, self_model_path[pth_name]))
                    requests.get(request_url[pth_name], {'weights_path':self_model_path[pth_name]})
                else:
                    print('TTS.py :input %s:%s is not available'%(pth_name, self_model_path[pth_name]))
        else:
            print("TTS.py:TTS未开启，请打开TTS接口，并将self.open_tts设为True") 
        
    def modify_params(self, params={}):
        self.params = self.default_params
        if params:  
            for key,val in params.items():
                if val:
                    self.params[key] = val
                    if key in ['ref_audio_path', 'prompt_text']:
                        self.params[key] = self.params[key].strip('\'').strip('"')
        print("TTS.py:更新tts_params为：", self.params)
        
    def modify(self, model_path, params):
        '''当输入的参数存在空值时，用默认值代替'''
        if self.open_tts:
            self.modify_model(model_path)
            self.modify_params(params)
        else:
            print("TTS.py：TTS is closed, please open it firstly")
    
    def run(self, text):
        if self.open_tts:
            self.start_run(text)

    def start_run(self, text):
        self.params.update({"text":text})
        url = 'http://127.0.0.1:9880/tts'
        if 'streaming_mode' in self.params:
            out_stream = self.params['streaming_mode']
        else:
            out_stream = False
        if not out_stream:
            result = requests.post(url, json = self.params).content
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

        print("TTS.py：tts.run正在读取：", self.params['text'])
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
            for chunk in requests.post(url, json = self.params):
                if self.play:
                    stream.write(chunk)
                if self.audio_save_file:
                    wf.writeframes(chunk)
            if self.play:
                stream.stop_stream()#暂停
                stream.close()#关闭
                p.terminate()


    def tts_text(self, process_json):
        '''
        procession_json:{chatbot:, chat_end_flag:};
        chatbot是转为音频的文字，chat_end_flag是chatbot是否的结束的标志；
        根据chatbot输入的文本，根据endure_sentence，分段依次进行音频转写；
        如果chat_end_flag=1及chatbot生成结束，则将chatbot待转换的文本一并送去转写为音频;输出tts转换完成标识，当tts开启时，返回1，否者返回0。
        '''
        
        if self.open_tts:
            self.tts_text_open(process_json)
            return(1)
        elif process_json['chat_end_flag']==1:
            with open(self.cache_path+'tts_have_read_length.txt','r') as file:
                temp_val = file.read()
            ##chat_end_flag=1表示对话完成，直接传入chatbot最后回复的长度
            #“['chatbot'][-1]”不能动，动了就不对了！！
            try:
                temp_val = len(process_json['chatbot'][-1][1])
                open(self.cache_path+'tts_have_read_length.txt','w+').write(str(temp_val))
            except:
                pass
        return(0)
    def tts_text_open(self, process_json):
        num = process_json['chat_end_flag']
        history = process_json['chatbot']
        have_read_length =0
        with open(self.cache_path+'tts_have_read_length.txt','r') as file:
            temp_val = file.read()
        if history:
            reply = history[-1][1]
        else:
            reply = ''
        if temp_val:
            have_read_length = int(temp_val)
        if self.display:
            print("TTS.py：传入tts_text的history:",history, "num:", num, "have_read_length", have_read_length,)
        if reply[have_read_length:]: 
            if self.display:
                print("TTS.py：tts.tts_text目前待读入的文字：，",reply[have_read_length:],'\n')
            reply = reply[have_read_length:]
            text_list = direct_cut(reply, self.splits)
            if self.display:
                print("TTS.py：tts.tts_text切分后的列表：，", text_list, "num:", num, "have_read_length:", have_read_length, "history的长度", len(history[-1][1]), '\n')
            if ((reply[-1] in self.splits)&(len(text_list) >=self.endure_sentence))|(num==1):
                if self.display:
                    print("TTS.py：tts.tts_text待读入文本末尾是标点符号且文本长度大于endure_sentence<或者num=1且文本长度非0，进入语音转写的循环")
                while ((num==1)&(len(text_list)>0))|(len(text_list) >=self.endure_sentence):
                    if num==1:
                        final_text = ''.join(text_list)
                    else:
                        final_text = ''.join(text_list[:self.endure_sentence])
                    #如果进入循环，需要把text_list的内容读完，所以先提前设定have_read_length防止后面chatbot或者num改变的时候重复读取
                    have_read_length += len(final_text)
                    if self.display:
                        print("TTS.py：tts.tts_text截止到下一步要读取的总长度%d"%have_read_length)
                    temp_val = have_read_length
                    #下面这段代码不能删除，否者回复tts_have_read_length不归0
                    open(self.cache_path+'tts_have_read_length.txt','w+').write(str(temp_val)) 
                    if final_text.strip():
                        # if self.display:
                        #     print("TTS.py：tts.tts_text将要读取的文字：", final_text,'\n')
                        self.run(final_text)
                        # if self.display:
                        #     print("TTS.py：tts.tts_text将文字“%s”读取完成！\n"%final_text)
                    if num!=1:
                        text_list = text_list[self.endure_sentence:]
                    else:
                        text_list = []
                    if self.display:
                        print("TTS.py：tts.tts_text文字“%s”读取完成后的text_list:"%final_text, text_list)
                        print("目前已经读入的文字长度：%d"%have_read_length)
        #下面这句话不能写，否者切换记录会复读
        # open(self.cache_path+'other_params_saved.json','w+').write(json.dumps(temp_val))    

    def modify_endure_sentence(self, num):
        self.endure_sentence = num
        print("TTS.py：tts endure_sentence更改为%d"%num)

#----------------------------------------------------------------------------------------#

def params_overall_json_update(json, gpt_path,sovits_path,ref_audio_path, prompt_text, batch_size,prompt_lang,text_lang, seed,text_split_method,speed_factor):
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
    json = gr.Json(value=json)
    return(json)
    
def direct_cut(sentence, splits):
    '''将sentence根据splits切分为多段'''
    char_list = ['']
    for index in range(len(sentence)):
        char1 = sentence[index]
        char_list[-1] += char1
        if char1 in splits:
            char_list.append('')
    if not char_list[-1]:
        char_list = char_list[:-1]
    return(char_list)
        



    