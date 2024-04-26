import torch
import ollama
import os,re
import subprocess
import json
import time
from tts import TTS
from py_run_cmd import run_cmd
from save_params import init_chat_params
rc = run_cmd()
print("model.py中rc path的位置%s"%os.path.abspath(rc.path))
messages =[]
#文本处理

class Model():
    def __init__(self, path, messages=messages, disply_inf =False):
        '''
        path:cache_file的路径
        messages:对话记录
        '''
        self.path = path
        self.messages =messages
        #display_params_infomation:default=True
        self.disply_inf =disply_inf
        self.model, self.model_loader, self.promot_text, self.cut_params, self.params="None", 'None', '',  {},{}
        self.use_tts = False
        self.splits = {'\n', ',','.','。','，','!','！', '?', '？',}
    '''
    model.run没进行一次，都读取本地文件，更新变量(可能影响效率)
    cut_params:
        'max_epoch':25,
        'cut_nepoch':8,
    ollama--params:
        'num_predict': 512,#生成的最大tokens数,default:128
        'temperature': 0.7,#default:0.8
        'top_p': 0.9,
        'top_k':20#defalut:40
        'seed':42,default:0
        'num_ctx':2048,#default=2048
        'num_gpu':6,#使用gpu运行模型的层数，num_gpu=0时不使用gpu，为1的时候使用gpu内存。
        'repeat_penalty':1.2, #default=1.1
        'stop':["AI assistant:"],#对话停止的输入
    '''
    def params_update(self):
        '''
        参数文件初始化及处理，如果找不到文件，则创建一个默认文件
        '''
        temp_dict = init_chat_params(self.path, 'chat_params.json')
        self.model, self.model_loader, self.cut_params, params, self.promot_text = temp_dict['model'], temp_dict['model_loader'], temp_dict['cut_params'], temp_dict['params'], temp_dict['promot_text']

        # print(params)

        #参数初始化
        self.params={}
        if (self.model_loader=='ollama')&('ollama' in params):
            #启动ollama app--ollama无法更改路径，所以固定
            if not rc.find_cmd_pid('ollama app'):
                print("在Model.py文件中开启ollama服务")
                exe_path = os.path.expanduser('~')+'\\AppData\\Local\\Programs\\Ollama\\ollama app.exe'
                subprocess.Popen([exe_path],stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_CONSOLE)
            self.params['num_predict'] =2**params['ollama']['num_predict']
            self.params['num_ctx'] = 2**params['ollama']['num_ctx']
            if type(params['ollama']['stop']) !=list:
                self.params['stop'] = params['ollama']['stop'].split('\n')
            if "num_gpu" in params['ollama']:
                if params['ollama']['num_gpu'] !=-1:
                    self.params['num_gpu'] = params['ollama']['num_gpu']
            for key in params['ollama']:
                if key not in ['num_predict', 'num_ctx', 'stop', 'num_gpu']:
                    self.params[key] = params['ollama'][key]
        if self.disply_inf:
            print(f"\nModel params update\ncheck_init_params:\n params:{self.params},\ncut_params:{self.cut_params},\nmodel:{self.model},\nmodel_loader:{self.model_loader},\npromot_text:{self.promot_text}")

    def add_promot(self):
        #判断与上一个promot是否相同
        self.params_update()
        sys_message = [message['content'] for message in self.messages if message['role']=='system']
        if (self.model_loader=='ollama'):
            if len(sys_message)==0:
                if self.promot_text:
                    self.messages.append({'role':'system','content':self.promot_text})
                    if self.disply_inf:
                        print(f'curr_model_loader:{self.model_loader}, add promot text:{self.promot_text}')
            elif sys_message[-1]!=self.promot_text:
                if self.promot_text:
                    self.messages.append({'role':'system','content':self.promot_text})
                    if self.disply_inf:
                        print(f'curr_model_loader:{self.model_loader}, add promot text:{self.promot_text}')

    def clear(self):
        self.messages=[]
        self.add_promot()
        if self.disply_inf:
            print('clear messages record!')

    def load_messages(self):
        name = 'None'
        # 将chat_record显示的值放入到other_params_saved.json里面
        if  os.path.exists(self.path+'other_params_saved.json'):
            with open(self.path+'other_params_saved.json','r') as file:
                temp_val = json.loads(file.read())
                if 'record_time' in temp_val:
                    temp_val_1 = temp_val['record_time'].strip('\n')
                    if temp_val_1:
                        name = temp_val_1[-1]

        to_loader_messages=dict()
        if os.path.exists(self.path+'history.json'):
            with open(self.path+'history.json','r') as file:
                temp_val = file.read()
                if temp_val:
                    to_loader_messages = json.loads(temp_val)
        if (name !="None") and (name in to_loader_messages):
            self.messages = to_loader_messages[name][1]
            print(f"load messages from record, messages is: \n{self.messages}") 

    def load_history_json_file(self):
        '''
        history_jon_file内容格式：
        {key:[history,messages]}, 其中history:[[问，答]，[问，答],..]，messages：[{'role':.., 'cotent':..},..]
        '''
        history=dict()
        if os.path.exists(self.path+'history.json'):
            with open(self.path+'history.json','r') as file:
                temp_val = file.read()
                if temp_val:
                    history = json.loads(temp_val)
        return(history)

    def return_history(self):
        self.params_update()
        print(f'model messages is\n{self.messages}')
        if self.model_loader=="None":
            return([])
        elif self.model_loader =='ollama':
            temp = [temp['content'] for temp in self.messages if temp['role'] in ['user', 'assistant']]
            if len(temp)==0:
                return([])
            elif len(temp)%2==1:
                temp.append(None)
            return(list(map(lambda x:list(x), zip(temp[::2], temp[1::2]))))
    
    def save_history_to_cache(self):
        history=dict()
        if os.path.exists(self.path+'history.json'):
            with open(self.path+'history.json','r') as file:
                temp_val = file.read()
                if temp_val:
                    history = json.loads(temp_val)
        #history.json里面数据格式：{key:[history,messages]}, 其中history:[[问，答]，[问，答],..]，messages：[{'role':.., 'cotent':..},..]
        history.update({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()):[self.return_history(), self.messages]})
        with open(self.path+'history.json','w+') as file:
            file.write(json.dumps(history))
        # return(history)
        

    def bot(self, history):
        '''
        when model_loader==None, output fix words
        '''
        # print(f'test!!!!!!!!!!!!:此轮对话前的messages为\n{self.messages}')
 
        if self.disply_inf:
            print('history:', history)
        # self.add_promot()

        if self.model_loader =='None':
            history[-1][1]=''
            for i in "hello, you don't choose available model, please choose model left.":
                history[-1][1] +=i
                time.sleep(0.05)
                yield history
        elif self.model_loader=='ollama':
            #对话裁剪
            if self.cut_params:
                epoch = [1 for message in self.messages if message['role']=="assistant"]
                if len(epoch) >= self.cut_params['max_epoch']:
                    self.cut_history()
            self.messages.append({"role": "user", "content": history[-1][0]})
            stream = ollama.chat(
            model=self.model,
            messages=self.messages,
            stream=True,
            #option中的参数含义：https://github.com/ggerganov/llama.cpp/tree/master/examples/main#number-of-tokens-to-predict
            options=self.params,
            )
            self.messages.append({'role': 'assistant','content':''})
            history[-1][1] = ""
            if self.use_tts:
                words = ''
            for chunk in stream:
                content = chunk['message']['content']
                content = re.sub('\n+','\n', content) 
                if content:
                    if self.use_tts:
                        words += content
                        if content in self.splits:
                            self.tts.run(words)
                            words=''
                    history[-1][1] += content
                    self.messages[-1]['content'] += content
                    yield history
        
    
    def get_system(self,):
        cur_system = [self.messages['content'] for self.messages in messages if self.messages['role']=="system"]
        if cur_system:
            cur_system = cur_system[-1]
        else:
            cur_system =''
        return(cur_system)
    
    def cut_history(self):
        curr_system = self.get_system()
        #定义裁剪的轮速
        now_num = 1
        mess_ind =0
        while( now_num <= self.cut_params['cut_nepoch']):
            if self.messages[mess_ind]['role'] == 'assistant':
                now_num +=1
                mess_ind += 1
                continue
            mess_ind += 1
        
        self.messages = self.messages[mess_ind:].copy()
        #如果system的记录被删除了，则在开头加上
        if (not self.get_system())&(curr_system !=''):
            self.messages.insert(0, {'role':'system','content':curr_system})

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    
    def add_tts(self, model_path={}, params={}, play=None, audio_save_file=None, tts_api_path=None):
        print('-'*20)
        print("当前TTS参数为：")
        print("model_path:", model_path)
        print("params", params)
        print('-'*20)
        # self.use_tts=True
        # self.tts = TTS()
        if model_path:
            self.tts.modify_model(model_path)
        if params:
            self.tts.modify_params(params)
        if play is not None:
            self.tts.play =play
        if audio_save_file is not None:
            self.tts.audio_save_file = audio_save_file
        if tts_api_path is not None:
            self.tts.tts_api_path = tts_api_path
        
    def dismiss_tts(self):
        print('close tts')
        self.tts = None
        self.use_tts=False
    
    def enable_tts(self):
        self.use_tts=True
        self.tts = TTS(model_path={})