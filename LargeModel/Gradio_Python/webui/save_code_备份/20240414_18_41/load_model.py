import gradio as gr
import ollama
def found_model_loader():
    #待办：model_loader待传入json
    model_loader = ['None', 'ollama']
    model_loader = gr.Dropdown(model_loader, label='model_loader')
    return(model_loader)

def found_model():
    model = gr.Dropdown(['None'], label='model')
    return(model)

def model_select_update(model_loader, model):
    # model_loader与model的级联界面，借鉴自https://zhuanlan.zhihu.com/p/663411336
    if model_loader=='None':
        models_list = ['None']
    elif model_loader =='ollama':
        #待办：可以事先写入一个json文件
        models_list = ['None']+[model['name'] for model in  ollama.list()['models']]
    model = gr.Dropdown.update(choices=models_list)
    return(model)

def save_model(model):
    with open(r'.\cache\model_val.txt', 'a+', encoding='utf-8') as file:
        file.write(model + "\n") 
    return

def save_model_loader(model_loader):
    with open(r'.\cache\model_loader_val.txt', 'a+', encoding='utf-8') as file:
        file.write(model_loader + "\n") 
    return

def save_promot_text(promot):
    with open(r'.\cache\promot_text.txt', 'w+') as file:
        file.write(promot) 
    return




    


