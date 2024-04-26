import torch
import ollama
import os,re
import json
import time
messages =[]
#文本处理

class Model():
    def __init__(self, path, messages=messages):
        '''
        path:cache_file的路径
        messages:对话记录
        '''
        self.path = path
        self.messages =messages
        #display_params_infomation:default=True
        self.disply_inf =True
    '''
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
        if not os.path.exists(self.path+'promot_text.txt'):
            with open(self.path+'promot_text.txt', 'w+') as file:
                file.write('')
                self.promot_text =''
        else:
            with open(self.path+'promot_text.txt', 'r') as file:
                temp_val = file.read().strip()
                if temp_val:
                    self.promot_text = temp_val
                else:
                    self.promot_text =''
    
        if not os.path.exists(self.path+'model_loader_val.txt'):
            with open(self.path+'model_loader_val.txt', 'a+', encoding='utf-8') as file:
                file.write('None\n')
                self.model_loader='None'
        else:
            with open(self.path+'model_loader_val.txt','r', encoding='utf-8') as file:
                temp_val = file.read().strip('\n')
                if temp_val:
                    self.model_loader = temp_val.split('\n')[-1]
                else:
                    self.model_loader='None'

        if not os.path.exists(self.path+'model_val.txt'):
            with open(self.path+'model_val.txt', 'a+', encoding='utf-8') as file:
                file.write('None\n')
                self.model = "None"
        else:
            with open(self.path+'model_val.txt','r', encoding='utf-8') as file:
                temp_val = file.read().strip('\n')
                if temp_val:
                    self.model = temp_val.split('\n')[-1]
                else:
                    self.model = "None"

        if not os.path.exists(self.path+'cut_params.json'):
            with open(self.path+'cut_params.json', 'w+') as file:
                file.write('{}')
                self.cut_params=dict()
        else:
            with open(self.path+'cut_params.json','r') as file:
                temp_val = file.read()
                if temp_val:
                    self.cut_params = json.loads(temp_val)
                else:
                    self.cut_params=dict()

        if not os.path.exists(self.path+'params.json'):
            with open(self.path+'params.json', 'w+') as file:
                file.write('{}')
                self.params=dict()
        else:
            with open(self.path+'params.json','r') as file:
                temp_val = file.read()
                if temp_val:
                    self.params = json.loads(temp_val)
                else:
                    self.params=dict()
                #参数初始化
        if self.model_loader=='ollama':
            self.params['num_predict'] =2**self.params['num_predict']
            self.params['num_ctx'] = 2**self.params['num_ctx']
            self.params['stop'] = self.params['stop'].split('\n')
            if self.params['num_gpu'] ==-1:
                del self.params['num_gpu']
        if self.disply_inf:
            print(f"\ncheck_init_params:\n params:{self.params},\ncut_params:{self.cut_params},\nmodel:{self.model},\nmodel_loader:{self.model_loader},\npromot_text:{self.promot_text}\n")

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

    def bot(self, history):
        '''
        when model_loader==None, output fix words
        '''
        if self.disply_inf:
            print('history:', history)
        self.add_promot()
        if self.model_loader =='None':
            history[-1][1]=''
            for i in "hello, you don't choose available model, please choose model left.":
                history[-1][1] +=i
                time.sleep(0.02)
                yield history
        elif self.model_loader=='ollama':
            #对话裁剪
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
            for chunk in stream:
                content = chunk['message']['content']
                content = re.sub('\n+','\n', content) 
                if content:
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
            