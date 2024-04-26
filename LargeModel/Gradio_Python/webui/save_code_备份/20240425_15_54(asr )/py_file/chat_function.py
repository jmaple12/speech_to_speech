import gradio as gr
import json
import os
#根据run.bat的相对位置
cache_path='./cache/'


def clear_inform(chatbot):
    chatbot = gr.Chatbot([])
    return(chatbot)

def json_append(json_1,json_2):
    json_2(json_1)
    json_2 = gr.Json(value=json_2)
    return(json_2)

def json_default():
    history=dict()
    if os.path.exists(cache_path+'history.json'):
        with open(cache_path+'history.json','r') as file:
            temp_val = file.read()
            if temp_val:
                history = json.loads(temp_val)
    return(history, sorted(history.keys()))

def json_update(json_2):
    history=dict()
    if os.path.exists(cache_path+'history.json'):
        with open(cache_path+'history.json','r') as file:
            temp_val = file.read()
            if temp_val:
                history = json.loads(temp_val)

    json_2 = gr.Json(value=history)
    return(json_2)

def update_record(save_chat_json2, chat_record):

    index = sorted(save_chat_json2.keys())
    chat_record = gr.Dropdown(choices=index)
    return(chat_record)

def delete_record(chat_record, save_chat_json):
    '''删除chat_record的当前值并将save_chat_json存储到history.json'''
    if chat_record in save_chat_json:
        del save_chat_json[chat_record]
    #将json 存储到cache里面的history.json
    with open(cache_path+'history.json','w+') as file:
        file.write(json.dumps(save_chat_json))
    save_chat_json = gr.Json(value=save_chat_json)
    return(save_chat_json)


def display_record_chat(chat_record, save_chat_json, chatbot):
    if chat_record in save_chat_json:
        chatbot = gr.Chatbot(save_chat_json[chat_record][0])
        print()
    return(chatbot)

def model_load_weight(chat_record):
    if os.path.exists(cache_path+'other_params_saved.json'):
        with open(cache_path+'other_params_saved.json', 'r') as file:
            temp_val = json.loads(file.read())
            if 'record_time' not in temp_val:
                temp_val['record_time'] = ''
            if chat_record not in temp_val['record_time']:
                temp_val['record_time'] += (chat_record+'\n')
        open(cache_path+'other_params_saved.json', 'w+').write(json.dumps(temp_val))
    else:
        open(cache_path+'other_params_saved.json', 'w+').write(json.dumps({'record_time':'None\n'}))



    


