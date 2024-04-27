
import gradio as gr
import ollama
import subprocess
import os
import json

#启动ollama app
#ollama app路径：r"C:\Users\maple\AppData\Local\Programs\Ollama\ollama app.exe"
exe_path = os.path.expanduser('~')+'\\AppData\\Local\\Programs\\Ollama\\ollama app.exe'
subprocess.Popen([exe_path],stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_CONSOLE)

print("load_model.py从%s启动ollama app.exe"%exe_path)
#提前加载防止控件卡顿
ollama_list = ollama.list()['models']
#根据run.bat的相对位置
cache_path='./cache/'
def found_model_loader():
    #待办：model_loader待传入json
    model_loader = ['None', 'ollama']
    model_loader = gr.Dropdown(model_loader, label='model_loader', value="None")
    return(model_loader)

def found_model():
    model = gr.Dropdown(['None'], label='model', value="None")
    return(model)

def model_select_update(model_loader, model):
    # model_loader与model的级联界面，借鉴自https://zhuanlan.zhihu.com/p/663411336
    if model_loader=='None':
        models_list = ['None']
    elif model_loader =='ollama':
        #待办：可以事先写入一个json文件
        models_list = ['None']+[model['name'] for model in ollama_list]
    model = gr.Dropdown(choices=models_list)
    return(model)

def save_model(model):
    chat_params=dict()
    if os.path.exists(cache_path+'chat_params.json'):
        with open(cache_path+'chat_params.json','r') as file:
            temp = file.read()
            if temp:
                chat_params = json.loads(temp)
    chat_params['model'] = model+'\n'
    with open(cache_path+'chat_params.json','w+') as file:
        file.write(json.dumps(chat_params))
    return

def save_model_loader(model_loader):
    chat_params=dict()
    if os.path.exists(cache_path+'chat_params.json'):
        with open(cache_path+'chat_params.json','r') as file:
            temp_val = file.read()
            if temp_val:
                chat_params = json.loads(temp_val)
    chat_params['model_loader'] = model_loader+'\n'
    with open(cache_path+'chat_params.json','w+') as file:
        file.write(json.dumps(chat_params))
    print("model_loader saved", chat_params)
    return

def save_promot_text(promot):
    chat_params=dict()
    if os.path.exists(cache_path+'chat_params.json'):
        with open(cache_path+'chat_params.json','r') as file:
            temp = file.read()
            if temp:
                chat_params = json.loads(temp)
    chat_params['promot_text'] = promot+'\n'
    with open(cache_path+'chat_params.json','w+') as file:
        file.write(json.dumps(chat_params))
    return




    


