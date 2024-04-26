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
        if not os.path.exists(self.params_path+'promot_text.txt'):
            promot_text =''
        else:
            with open(self.params_path+'promot_text.txt', 'r') as file:
                temp_val = file.read().strip()
                if temp_val:
                    promot_text = temp_val
                else:
                    promot_text =''

        if not os.path.exists(self.params_path+'model_loader_val.txt'):
            model_loader='None'
        else:
            with open(self.params_path+'model_loader_val.txt','r', encoding='utf-8') as file:
                temp_val = file.read().strip('\n')
                if temp_val:
                    model_loader = temp_val.split('\n')[-1]
                else:
                    model_loader='None'

        if not os.path.exists(self.params_path+'model_val.txt'):
            model = "None"
        else:
            with open(self.params_path+'model_val.txt','r', encoding='utf-8') as file:
                temp_val = file.read().strip('\n')
                if temp_val:
                    model = temp_val.split('\n')[-1]
                else:
                    model = "None"

        if not os.path.exists(self.params_path+'cut_params.json'):
            cut_params=dict()
        else:
            with open(self.params_path+'cut_params.json','r') as file:
                temp_val = file.read()
                if temp_val:
                    cut_params = json.loads(temp_val)
                else:
                    cut_params=dict()

        if not os.path.exists(self.params_path+'params.json'):
            params=dict()
        else:
            with open(self.params_path+'params.json','r') as file:
                temp_val = file.read()
                if temp_val:
                    params = json.loads(temp_val)
                else:
                    params=dict()
        return({'model':model, 'model_loader':model_loader, 'cut_params':cut_params, 'params':params, 'promot_text':promot_text})

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
            print('target file is not found')
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
            print('target file is not found')
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
            print('target file is not found')
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
            print('target file is not found')
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
        model = gr.Dropdown.update(value=model_params['model'])
        model_loader = gr.Dropdown.update(value=model_params['model_loader'])
        if model_params['cut_params']:
            max_epoch= gr.Slider.update(value=model_params['cut_params']['max_epoch'])
            cut_nepoch =gr.Slider.update(value=model_params['cut_params']['cut_nepoch'])
        if model_params['params']:
            num_predict = gr.Slider.update(value=model_params['params']['num_predict'])
            temperature = gr.Slider.update(value=model_params['params']['temperature'])
            top_p = gr.Slider.update(value=model_params['params']['top_p'])
            top_k = gr.Slider.update(value=model_params['params']['top_k'])
            num_ctx = gr.Slider.update(value=model_params['params']['num_ctx'])
            repeat_penalty = gr.Slider.update(value=model_params['params']['repeat_penalty'])
            seed = gr.Number.update(value=model_params['params']['seed'])
            num_gpu = gr.Number.update(value=model_params['params']['num_gpu'])
            stop = gr.Textbox.update(value=model_params['params']['stop'])
        promot_text = gr.Textbox.update(value=model_params['promot_text'])
        #将参数存储到cache---params_path存放处
        with open(ccp.params_path+'model_val.txt', 'a+', encoding='utf-8') as file:
            file.write(model_params['model'] + "\n") 
        with open(ccp.params_path+'model_loader_val.txt', 'a+', encoding='utf-8') as file:
            file.write(model_params['model_loader'] + "\n") 
        with open(ccp.params_path+'promot_text.txt', 'w+') as file:
            file.write(model_params['promot_text']) 
        with open(ccp.params_path+'cut_params.json','w+') as file:
            file.write(json.dumps(model_params['cut_params']))
        with open(ccp.params_path+'params.json','w+') as file:
            file.write(json.dumps(model_params['params']))
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
    saved_params_display_tex = gr.Textbox.update(value='\n'.join(val))
    return(saved_params_display, saved_params_display_copy, saved_params_display_tex)

def modify_label(save_params_button, refresh_params_dp):
    if refresh_params_dp=='cover':
        save_params_button = gr.Button.update('click to save current all params  to particular path, cover origin file')
    elif refresh_params_dp == 'add':
         save_params_button = gr.Button('click to add current all params to particular path')
    elif refresh_params_dp == 'refresh':
        save_params_button = gr.Button('click to refresh current saved params')
    return(save_params_button)

def params_modify_textbox(save_params_modify, saved_params_display,saved_params_display_copy, saved_params_display_tex):
    if save_params_modify and saved_params_display:
        ccp = create_cus_params()
        ccp.modify_saved_params(saved_params_display, save_params_modify)
        temp_str = saved_params_display_tex.replace(saved_params_display, save_params_modify)
        saved_params_display_tex = gr.Textbox.update(temp_str)
        if '\n' not in temp_str:
            temp_str = [temp_str]
        else:
            temp_str = temp_str.split('\n')
        saved_params_display = gr.Dropdown.update(choices=temp_str)
        saved_params_display_copy = gr.Dropdown.update(choices=temp_str)
    else:
        print("dont submit null string, or no params has been saved")
    return(saved_params_display, saved_params_display_copy, saved_params_display_tex)


def params_delete_button(saved_params_display,saved_params_display_copy, saved_params_display_tex):
    if saved_params_display:
        ccp = create_cus_params()
        ccp.delete_saved_params(saved_params_display)
        temp_str = saved_params_display_tex.replace(saved_params_display,'')
        temp_str = re.sub('\n+', '\n', temp_str).strip('\n')
        saved_params_display_tex = gr.Textbox.update(temp_str)
        if not temp_str:
            temp_str =[]
        elif '\n' not in temp_str:
            temp_str = [temp_str]
        else:
            temp_str = temp_str.split('\n')
        saved_params_display = gr.Dropdown.update(choices=temp_str)
        saved_params_display_copy = gr.Dropdown.update(choices=temp_str)
    else:
        print("no params has been saved, please add one before")
    return(saved_params_display, saved_params_display_copy, saved_params_display_tex)