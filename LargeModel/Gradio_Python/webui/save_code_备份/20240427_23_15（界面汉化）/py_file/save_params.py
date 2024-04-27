import os, re
import json
import gradio as gr
#根据run.bat的相对位置
cache_path='./cache/'

class create_cus_params():
    def __init__(self, saved_path=cache_path, params_path=cache_path):
        '''
        saved_path:path to save params
        params_path:current params path
        saved_path_exists:saved_path exists or not
        '''
        self.saved_path = saved_path.replace('\\','/')
        self.params_path = params_path
        if self.saved_path[-1] !='/':
            self.saved_path= self.saved_path+'/'
        
        self.path = self.saved_path+'save_params/params.json'
        if not os.path.exists(self.path):
            self.saved_path_exists = False
        else:
            self.saved_path_exists=True
        
    def return_curr_params(self):
        '''
        读取cache存储的参数文件，如果不存在，返回默认值
        '''
        return(init_chat_params(self.params_path, 'chat_params.json'))

    def save_curr_params(self, add=True):
        '''
        将当前的参数追加到自定义的文件夹
        add:是否追加数据，False的时候覆盖
        '''
        now_dir = os.getcwd()
        params_to_save = self.return_curr_params()
        os.chdir(os.path.abspath(self.saved_path[:-1]))
        if not os.path.exists('save_params'):
            os.makedirs('save_params')
        exists_params=dict()
        if add:
            if not os.path.exists('save_params/params.json'):
                with open('save_params/params.json', 'w+') as file:
                    file.write('{}')
            else:
                with open('save_params/params.json','r') as file:
                    temp_val = file.read()
                    if temp_val:
                        exists_params = json.loads(temp_val)
            num = 1
            if exists_params:
                temp_list = [int(dk[14:]) for dk in exists_params if 'custom_params_' in dk]
                if temp_list:
                    num = max(temp_list)+1
                else:
                    num=1
            exists_params['custom_params_'+str(num)] = params_to_save
            with open('save_params/params.json', 'w+') as file:
                    file.write(json.dumps(exists_params))
        else:
            with open('save_params/params.json', 'w+') as file:
                    file.write(json.dumps({'custom_params_1':params_to_save}))
        os.chdir(now_dir)
        self.saved_path_exists=True

    def display_params_name(self):
        if not self.saved_path_exists:
            print('when display saved params, target file is not found')
            return([])
        else:
            with open(self.path,'r') as file:
                temp_val = file.read()
                if temp_val:
                    params = json.loads(temp_val)
                    return(list(params.keys()))
                else:
                    return([])

    def display_params(self, name):
        if not self.saved_path_exists:
            print('when display saved params, target file is not found')
            return({})
        else:
            with open(self.path,'r') as file:
                temp_val = file.read()
                if temp_val:
                    params = json.loads(temp_val)
                    return(params[name])
                else:
                    return({})


    def modify_saved_params(self, old_name, new_name):
        if not self.saved_path_exists:
            print('when modify saved params, target file is not found')
        else:
            with open(self.path,'r') as file:
                temp_val = file.read()
                if temp_val:
                    params = json.loads(temp_val)
                    if old_name in params:
                        if new_name in params:
                            print('new name exists in params')
                        else:
                            params[new_name] = params.pop(old_name)
                            self.save_json(params)
                    else:
                        print('old_name is not exists')
                    
                else:
                    print('file is null')

            
    def delete_saved_params(self, name):
        if not self.saved_path_exists:
            print('when delete saved params, target file is not found')
        else:
            with open(self.path,'r') as file:
                temp_val = file.read()
                if temp_val:
                    params = json.loads(temp_val)
                    if name in params:
                        del params[name]
                        self.save_json(params)
                    else:
                        print('name is not exists')
                else:
                    print('file is null')

    def clear_saved_params(self):
        self.save_json({})

    def save_json(self, params):
        with open(self.path, 'w+') as file:
                file.write(json.dumps(params))   

    # def save_to_params_path(self, params):
    #     with open(temp, 'w+') as file:
    #             file.write(json.dumps(params))  


def display_params_button(saved_params_display, model, model_loader, 
                num_predict,temperature, top_p, top_k, num_ctx,repeat_penalty, seed, num_gpu,stop, max_epoch, cut_nepoch, promot_text):
    print('saved params display:', saved_params_display)
    if saved_params_display:
        ccp = create_cus_params()
        model_params = ccp.display_params(saved_params_display)
        model = gr.Dropdown(value=model_params['model'])
        model_loader = gr.Dropdown(value=model_params['model_loader'])
        if model_params['cut_params']:
            max_epoch= gr.Slider(value=model_params['cut_params']['max_epoch'])
            cut_nepoch =gr.Slider(value=model_params['cut_params']['cut_nepoch'])
        if model_params['params']:
            if "ollama" in model_params['params']:
                num_predict = gr.Slider(value=model_params['params']["ollama"]['num_predict'])
                temperature = gr.Slider(value=model_params['params']["ollama"]['temperature'])
                top_p = gr.Slider(value=model_params['params']["ollama"]['top_p'])
                top_k = gr.Slider(value=model_params['params']["ollama"]['top_k'])
                num_ctx = gr.Slider(value=model_params['params']["ollama"]['num_ctx'])
                repeat_penalty = gr.Slider(value=model_params['params']["ollama"]['repeat_penalty'])
                seed = gr.Number(value=model_params['params']["ollama"]['seed'])
                num_gpu = gr.Number(value=model_params['params']["ollama"]['num_gpu'])
                stop = gr.Textbox(value=model_params['params']["ollama"]['stop'])
        promot_text = gr.Textbox(value=model_params['promot_text'])
        #将参数存储到cache---params_path存放处
        with open(ccp.params_path+'chat_params.json', 'w+') as file:
            file.write(json.dumps(model_params))
            
    return(model, model_loader, 
                num_predict,temperature, top_p, top_k, num_ctx,repeat_penalty, seed, num_gpu,stop, max_epoch, cut_nepoch, promot_text)

def ccps(saved_params_display, saved_params_display_copy, saved_params_display_tex, refresh_params_dp):
    # ['add and cover params', 'add but not cover params', 'refresh params']
    ccptt = create_cus_params()
    add=True
    if refresh_params_dp == 'cover':
        add = False
    elif refresh_params_dp == 'add':
        add =True
    if refresh_params_dp != 'refresh':
        ccptt.save_curr_params(add=add)
    val = ccptt.display_params_name()
    saved_params_display = gr.Dropdown(choices=val)
    saved_params_display_copy = gr.Dropdown(choices=val)
    saved_params_display_tex = gr.Textbox(value='\n'.join(val))
    return(saved_params_display, saved_params_display_copy, saved_params_display_tex)

def modify_label(save_params_button, refresh_params_dp):
    if refresh_params_dp=='cover':
        #丢弃其他参数组合
        save_params_button = gr.Button('点击仅保存当前选中的参数')
    elif refresh_params_dp == 'add':
         save_params_button = gr.Button('点击保存选中的参数')
    elif refresh_params_dp == 'refresh':
        save_params_button = gr.Button('click to refresh current saved params')
    return(save_params_button)

def params_modify_textbox(save_params_modify, saved_params_display,saved_params_display_copy, saved_params_display_tex):
    if save_params_modify and saved_params_display:
        ccp = create_cus_params()
        ccp.modify_saved_params(saved_params_display, save_params_modify)
        temp_str = saved_params_display_tex.replace(saved_params_display, save_params_modify)
        saved_params_display_tex = gr.Textbox(temp_str)
        if '\n' not in temp_str:
            temp_str = [temp_str]
        else:
            temp_str = temp_str.split('\n')
        saved_params_display = gr.Dropdown(choices=temp_str)
        saved_params_display_copy = gr.Dropdown(choices=temp_str)
    else:
        print("dont submit null string, or no params has been saved")
    return(saved_params_display, saved_params_display_copy, saved_params_display_tex)


def params_delete_button(saved_params_display,saved_params_display_copy, saved_params_display_tex):
    if saved_params_display:
        ccp = create_cus_params()
        ccp.delete_saved_params(saved_params_display)
        temp_str = saved_params_display_tex.replace(saved_params_display,'')
        temp_str = re.sub('\n+', '\n', temp_str).strip('\n')
        saved_params_display_tex = gr.Textbox(temp_str)
        if not temp_str:
            temp_str =[]
        elif '\n' not in temp_str:
            temp_str = [temp_str]
        else:
            temp_str = temp_str.split('\n')
        saved_params_display = gr.Dropdown(choices=temp_str)
        saved_params_display_copy = gr.Dropdown(choices=temp_str)
    else:
        print("no params has been saved, please add one before")
    return(saved_params_display, saved_params_display_copy, saved_params_display_tex)



def init_chat_params(path, file_name='chat_params.json'):
    '''
    读取cache存储的参数文件，如果不存在，返回默认值
    '''
    promot_text =''
    if os.path.exists(path+file_name):
        with open(path+file_name, 'r') as file:
            temp_val = file.read()
            if temp_val:
                temp_val = json.loads(temp_val)
                if 'promot_text' in temp_val:
                    promot_text = temp_val['promot_text'].strip()

    model_loader='None'
    if os.path.exists(path+file_name):
        with open(path+file_name,'r') as file:
            temp_val = file.read()
            if temp_val:
                temp_val = json.loads(temp_val)
                if "model_loader" in temp_val:
                    model_loader = temp_val['model_loader'].strip('\n').split('\n')[-1]
    
    model='None'
    if os.path.exists(path+file_name):
        with open(path+file_name,'r') as file:
            temp_val = file.read()
            if temp_val:
                temp_val = json.loads(temp_val)
                if "model" in temp_val:
                    model = temp_val['model'].strip('\n').split('\n')[-1]

    cut_params=dict()
    if os.path.exists(path+file_name):
        with open(path+file_name,'r') as file:
            temp_val = file.read()
            if temp_val:
                temp_val = json.loads(temp_val)
                if "cut_params" in temp_val:
                    cut_params = temp_val['cut_params']

    params={"None":{}}
    if os.path.exists(path+file_name):
        with open(path+file_name,'r') as file:
            temp_val = file.read()
            if temp_val:
                temp_val = json.loads(temp_val)
                if 'params' in temp_val:
                    params = temp_val['params']
    return({'model':model, 'model_loader':model_loader, 'cut_params':cut_params, 'params':params, 'promot_text':promot_text})