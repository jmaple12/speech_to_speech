# -*- coding: utf-8 -*-
import random
import gradio as gr
import os, re
import time
import json
from dotenv import load_dotenv, find_dotenv
import requests
# import asyncio
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from model import Model
from load_model import found_model, found_model_loader, model_select_update, save_model,save_model_loader,save_promot_text
from model_params import gr_cut_params,gr_model_params,cut_epoch,save_params,save_cut_params
from save_params import create_cus_params,display_params_button,ccps,params_modify_textbox, params_delete_button,modify_label
from chat_function import clear_inform, json_default, json_update, display_record_chat,model_load_weight, delete_record , update_record
from tts import params_overall_json_update
from py_run_cmd import run_cmd
from tts import TTS
from asr import asr_request,check_asr, asr_init,get_asr_process
webui_path = os.path.abspath('../webui')
cache_path = os.path.abspath('../webui/cache/').replace('\\','/')+'/'
asr_api_path = "E:\LargeModel\kaldi\sherpa_onnx_model\sherpa_onnx_speech_recognizier.bat"
#是否展示一些详细日志
display=False
_ = load_dotenv(find_dotenv())
def fake_gan(prompt):
    data['prompt'] = prompt
    response = requests.post(url,headers=headers,json=data)
    res = response.json()
    images = [
        (random.choice(res['result']['imageUrls']), f"U 0")
    ]
    return images
def generate_images_with_prompts(prompt, navigator_prompt, model_choice):

    return fake_gan(prompt)
def select_model(model_choice):
    return fake_gan()

#用户输入的文本
def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def add_file(history, file):
    history = history + [((file.name,), None)]
    return history

def dropdown_copy(origin, copy):
    if origin:
        copy = gr.Dropdown(value=origin)
    return(copy)

def curr_time():
    return(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

def display_hide_audio_params(use_tts):
    if use_tts == "click to start tts":
        return{tts_over_all_params:gr.Column(visible=True)}
    else:
        return{tts_over_all_params: gr.Column(visible=False)}

def run_close_tts(use_tts, tts_bat_path_upload):
    if use_tts == "click to start tts": 
        if tts_bat_path_upload:
            tts_bat_path_upload = tts_bat_path_upload.strip('"').strip('\'')
            rc.gpt_tts_run(tts_bat_path_upload)
        else:
            rc.gpt_tts_run()
        
        print('use_tts:', True)
        tts_funcion.open_tts=True

    else:
        print('use_tts:', False)
        rc.gpt_tts_close()
        tts_funcion.open_tts =False


def tts_process_json(target, tts_json, text):
    tts_json.update({text:target})
    if display:
        print("因为%s的改变，tts_proess_json更新为："%text,tts_json)
    return(tts_json)

def refresh_chat_end_flag(chatbot):
    temp_val = 0
    if chatbot:
        temp_val = len(chatbot[-1][1])
    if display:
        print('refresh_chat_end_flag', temp_val)
    open(cache_path+'tts_have_read_length.txt','w+').write(str(temp_val))

def check_tts(cache_path=cache_path):
    if not os.path.exists(cache_path +'tts_error.txt'):
        print("tts api未正常启动")
        return
    #每秒检查一次tts_error文件是否有地址
    for i in range(60):
        time.sleep(1)
        if re.search("http://\d+\.\d+\.\d+\.\d+", open(cache_path+'tts_error.txt','r', encoding='gb18030').read()):
            print("TTS API has launched")
            if os.path.exists(cache_path+'tts_params.json'):
                tts_params_file = json.loads(open(cache_path+'tts_params.json','r').read())
                tts_funcion.modify(tts_params_file['model_path'], tts_params_file['params'])
                # print("check_tts传入tts_params参数：", tts_params_file)
            else:
                print("tts传入默认参数")
            return(gr.Button("TTS launched"))
    print("api未开启")
    return


def loads_tts_bat_path_text(tts_bat_path_text1):
    
    temp_val = json.loads(open(cache_path+'other_params_saved.json','r').read())
    temp_val.update({'tts_bat':tts_bat_path_text1})
    if tts_bat_path_text1:
        print("更新tts_api文件位置为%s"%tts_bat_path_text1)
    else:
        print("更新tts_api文件位置为默认值")
    open(cache_path+'other_params_saved.json','w+').write(json.dumps(temp_val))

#计时器
def t_count(max_time=10**5):
    #默认最大计时间上限
    if not max_time:
        max_time=10**5
    for ttime in range(1,max_time):
        time.sleep(1)
        yield(ttime)

user_avatar = webui_path+"\\assets/user.png"
chatot_avatar = webui_path+"\\assets/chatbot.png"
user_avatar = user_avatar.replace('/','\\')
chatot_avatar = chatot_avatar.replace('/','\\')

#存储tts_have_read_length的文件
if not os.path.exists(cache_path+'tts_have_read_length.txt'):
    open(cache_path+'tts_have_read_length.txt', 'w+').write('0')


#存储tts_api_bat路径的文件，作为webui中tts_api_bat的初始值
if not os.path.exists(cache_path+"other_params_saved.json"):
    open(cache_path+'other_params_saved.json', 'w+').write(json.dumps({}))
other_params_flag=0 
with open(cache_path+'other_params_saved.json', 'r') as file:
    temp_val = file.read()
    if temp_val:
        temp_val =json.loads(temp_val)
        if 'tts_bat' in temp_val:
            if temp_val['tts_bat']:
                tts_bat_path_text_cache = temp_val['tts_bat']
                print("读取tts_api的文件路径为%s"%temp_val['tts_bat'])
                other_params_flag=1
    if other_params_flag==0:
        tts_bat_path_text_cache = ''
        print("读取tts_api默认文件位置")

#保存参数组合的管理系统：myccp-
myccp = create_cus_params()
#模型：model_run
model_run = Model(cache_path, disply_inf=True)
if model_run.use_tts:
    model_run.add_tts()
bot = model_run.bot
#python模拟cmd系统
rc = run_cmd()
#重启ollama
rc.clear_rerun_ollama()
#tts系统
tts_funcion = TTS(cache_path = cache_path, display=display)
tts_funcion.open_tts=False

with gr.Blocks() as demo:
    # ----------------------------------------------------------#
    # --------------------Chat Params--------------------------#
    # ----------------------------------------------------------#
    with gr.Tab('Params'):
        with gr.Row():
            #保存参数
            with gr.Column(min_width=200):
                #展示保存的参数字典--展示键值----待办：仅仅刷新页面并不能更新mycpp（后台没更新，所以saved_params_display = gr.Dropdown.并没有重新运行）
                saved_params_display = gr.Dropdown(label='saved_params_display', choices=myccp.display_params_name())
                #把键值存储到string里面
                saved_params_display_tex = gr.Textbox(visible=False)
                #字典值重命名
                save_params_modify = gr.Textbox(label='choose key name, input new name and then modify')
            with gr.Column(min_width=300): 
                # 字典值删除    
                save_params_delete = gr.Button('click to delete key name')
                with gr.Row():
                    #保存参数按钮save_params_button
                    save_cus_params_button = gr.Button('click to add current all params to particular path')
                    #对保存参数按钮增加几个选项---1。保存参数+覆盖源文件，2。保存参数+不覆盖源文件，3。刷新参数
                    refresh_params_dp = gr.Radio(label='save params with add or cover', choices=['add', 'cover'], value='add')
            
            restart_ollama = gr.Button('click to restart ollama app')

        with gr.Row():
            #system_pormot
            with gr.Column(min_width=400):
                promot_text = gr.Textbox(label='promot_text, enter to save', value="", lines=1)
                #提交动画：label:显示文本已经提交
                promot_text.submit(lambda x:gr.Textbox(label="text has submitted"), promot_text, promot_text, queue=True).then(lambda x:gr.Textbox(label="promot_text, enter to save"), promot_text, promot_text,)
                with gr.Row():
                    #选择框架
                    model_loader = found_model_loader()
                    #选择模型
                    model = found_model()
                #save_messages_cut_params button
                save_cut_param_button = gr.Button('click to save cut_params', visible=False)
                #点击动画：
                save_cut_param_button.click(lambda x:gr.Button('saved cut_params'), save_cut_param_button,save_cut_param_button, queue=True).then(lambda x:gr.Button('click to save cut_params'),save_cut_param_button,save_cut_param_button)
                # #选择最大对话轮次，超过时裁剪对话记录
                max_epoch, cut_nepoch = gr.Slider(visible=False),gr.Slider(visible=False)

            with gr.Column(min_width=100, scale=1):
                #将参数值存储cache/params.json
                save_model_loader_params_button = gr.Button('click to save model_loader params', visible=False)
                #点击动画：
                save_model_loader_params_button.click(lambda x:gr.Button('saved model_loader_params'), save_model_loader_params_button,save_model_loader_params_button, queue=True).then(lambda x:gr.Button('click to save model_loader params'), save_model_loader_params_button, save_model_loader_params_button)
                #选择模型参数---待办：参数用params代替
                with gr.Blocks():
                    num_predict = gr.Slider(visible=False)
                    temperature = gr.Slider(visible=False)
                    top_p = gr.Slider(visible=False)
                    top_k = gr.Slider(visible=False)
                    num_ctx = gr.Slider(visible=False)
                    repeat_penalty = gr.Slider(visible=False)
                    with gr.Row():
                        seed = gr.Number(visible=False)
                        num_gpu = gr.Number(visible=False)
                    stop = gr.Textbox(visible=False)
                
    #--------------------------Chat 界面---------------------------------#
    with gr.Tab('Chat'):
        with gr.Row():
            with gr.Column(min_width=200):
                #刷新按钮
                # save_params_button_copy = gr.Button('refresh current saved params')
                #展示保存的参数字典--展示键值----仅仅刷新页面并不能更新mycpp--copy
                saved_params_display_copy = gr.Dropdown(label='saved_params_display_copy', choices=myccp.display_params_name())
                #固定值的按钮---代替字符串使用
                refresh_params_dp_copy = gr.Textbox(value='refresh', visible=False)

                #聊天记录保存
                #保存的聊天记录存储到save_chat_json(隐藏)
                save_chat_json = gr.Json({}, visible=False, value=model_run.load_history_json_file())
                #下拉框的额默认值为
                chat_record = gr.Dropdown(label='chat record',choices=sorted(model_run.load_history_json_file().keys()))
                #刷新聊天记录按钮
                # refresh_chat_record = gr.Button('refresh chat record')
                #删除聊天记录按钮
                delete_chat_record = gr.Button('delete this record')
                
                #tts
                #tts_bat_path
                ##gradio将文件放到temp中的一个路径，地址也是这个地址，所以不能cd..，因此不能用Uploadbutton
                tts_bat_path_text = gr.Textbox(label="input gpt_tts bat file",value=tts_bat_path_text_cache.strip('\'').strip('"') if tts_bat_path_text_cache else None)
                tts_bat_path_text.submit(loads_tts_bat_path_text, tts_bat_path_text, None)
                with gr.Row():                  
                    #版本2
                    use_tts_on2 = gr.Button("click to start tts",)
                    use_tts_close2 = gr.Button("tts is launching", visible=False) 
                #对话中，每次传入几句话给tts
                tts_endure = gr.Slider(minimum=1, maximum=20, step=1, value=1, label="modify tts_edure_sentence")
                tts_endure.release(tts_funcion.modify_endure_sentence, tts_endure, None)

                #当chatbot或者chat_end_flag变化时，将内容加入到json后，然后根据json内容统一单线程传入tts_text
                tts_read_process_json = gr.Json({'chatbot':[['','']], 'chat_end_flag':0}, visible=False)
                show_tts_params = gr.Button("click to open tts params")
                
            with gr.Column(min_width=600, scale=8, visible=True):
                #定义聊天区域
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    bubble_full_width=False,
                    #显示头像
                    avatar_images=(user_avatar, (chatot_avatar)),
                    scale=4
                )

                #判断文本是否生成完毕
                chat_end_flag = gr.Number(value=0, visible=False)
                with gr.Row():
                    txt = gr.Textbox(
                        scale=20,
                        show_label=False,
                        placeholder="Enter text and press enter to submit, shift+enter to newline",
                        container=False,
                        label="chat now",
                    )
                    ##清屏+保存聊天记录=开启新聊天
                    #清屏按钮
                    clear_button = gr.Button('clear_chat', scale=0)
                    #保存聊天记录按钮
                    save_chat_button = gr.Button('save_chat', scale=0)
                    
                # #点击“chat”的反应
                # commit_btn = gr.Button("chat",scale=1)

            #--------------------------TTS Pamras------------------------#
            #tts_参数
            # 初始化tts参数的json文件
            tts_params_default = {'model_path':{'gpt_path':None, 'sovits_path':None}}
            tts_params_default['params'] = {'ref_audio_path':None,'prompt_text':'', 'batch_size':4,'prompt_lang':'zh','text_lang':'zh','seed':0,'text_split_method':'cut5','speed_factor':1, }

            if os.path.exists(cache_path+'tts_params.json'):
                with open(cache_path+'tts_params.json','r') as tts_p_file:
                    tts_p_file = json.loads(tts_p_file.read())
                    # 根据tts_p_file更新tts_params_default的值
                    for mid_name in tts_params_default:
                        if mid_name in tts_p_file:
                            for name in tts_params_default[mid_name]:
                                if name in tts_p_file[mid_name]:
                                    tts_params_default[mid_name][name] = tts_p_file[mid_name][name]
                    
                print('TTS参数初始化，默认必要参数为：', tts_params_default)
            else:
                open(cache_path+'tts_params.json','w+').write(json.dumps(tts_params_default))

            with gr.Column(visible=False) as tts_over_all_params:
                with gr.Row():
                    gpt_path = gr.Textbox(label='upload gpt model(.ckpt)', visible=True, value=tts_params_default['model_path']['gpt_path'])
                    sovits_path = gr.Textbox(label='upload sovits model(.pth)', visible=True, value=tts_params_default['model_path']['sovits_path'])
                    ref_audio_path = gr.Textbox(label='upload ref audio', visible=True, value=tts_params_default['params']['ref_audio_path'])

                with gr.Row():
                    prompt_text = gr.Textbox(label='input refer audio words', value=tts_params_default['params']['prompt_text'])
                    audio_seed = gr.Number(label='seed', value=tts_params_default['params']['seed'])
                with gr.Row():    
                    batch_size = gr.Slider(minimum=1, maximum=20, step=1, value=tts_params_default['params']['batch_size'], label='batch_size')
                    #语速
                    speed_factor = gr.Slider(minimum=0.1, maximum=3, step=0.1, value=tts_params_default['params']['speed_factor'], label='speed_factor')

                with gr.Row():
                    #切分方式
                    text_split_method_choices = ["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切", "根据句末结束符(中英文。？！等)切"]
                    text_split_method_show = gr.Dropdown(text_split_method_choices, label='prompt_text', value= text_split_method_choices[int(tts_params_default['params']['text_split_method'][3:])])
                    text_split_method = gr.Dropdown(['cut'+str(i) for i in range(7)],label='prompt_text', value=tts_params_default['params']['text_split_method'], visible=False)            
                    text_split_method_show.select(lambda zh:gr.Dropdown(value="cut"+str([item[0] for item in enumerate(["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切", "根据句末结束符(中英文。？！等)切"]) if item[1]==zh][0])), text_split_method_show, text_split_method)
                    #promot_word language
                    prompt_lang = gr.Dropdown(['zh','en','ja'],label='prompt_text', value=tts_params_default['params']['prompt_lang'])
                    #文本语言
                    text_lang = gr.Dropdown(['zh','en','ja'], label='text_lang', value=tts_params_default['params']['text_lang'])
                
                #保存参数的按钮
                audio_params_submit = gr.Button('Save and Apply tts params')
                audio_params_json = gr.Json(value={'model_path':{}, 'params':{}}, visible=False)
                hide_params_button = gr.Button("click to hide tts params")
    
        #-----------------ASR--------------------------------------#
        #计时器
        time_count = gr.Number(0, visible=True, label='time count')
        ##开启asr按钮---打开
        #star_asr的值不能轻易改
        start_asr = gr.Button("start asr")
        end_asr = gr.Button("asr launched", visible=False)
        with gr.Row():
            asr_pid = gr.Textbox(visible=True, label="asr_pid")
            asr_url = gr.Textbox(visible=True, label="asr_url")
            asr_port = gr.Textbox(visible=True, label="asr_port")
            asr_url.change(lambda x:x.split(':')[-1] if ':' in x else None, asr_url, asr_port)
    asr_text_process = gr.Textbox(label="show asr process")
    
    #点击重启ollama
    restart_ollama.click(lambda :gr.Button("restarting ollama for 3 seconds"), None, restart_ollama, queue=True).then(rc.clear_rerun_ollama, None, None).then(lambda :gr.Button("click to restart ollama app"), None, restart_ollama)
    #存储promot_text
    promot_text.submit(save_promot_text, inputs=[promot_text], outputs=None, queue=True).then(model_run.add_promot, None, None)
    #model_loader当选择为ollama时，重启ollama app，else 关闭
    model_loader.select(lambda x:rc.close_ollama() if x!="ollama" else rc.clear_rerun_ollama(), model_loader, None)
    #存储model_loader值
    model_loader.change(save_model_loader, inputs=[model_loader], outputs=None)
    model_loader.change(model_run.params_update, None,None)
    #根据model_loader展示model值
    model_loader.change(model_select_update, inputs=[model_loader, model], outputs=[model])
    #根据model_loader选项打开cut_params
    model_loader.change(gr_cut_params, inputs=[model_loader, max_epoch, cut_nepoch,save_cut_param_button], outputs=[max_epoch, cut_nepoch, save_cut_param_button])
    #根据model_loader选项展示params中各变量
    model_loader.change(gr_model_params, inputs=[model_loader,save_model_loader_params_button, num_predict,temperature,top_p,top_k,num_ctx, repeat_penalty, seed,num_gpu,stop], outputs=[save_model_loader_params_button, num_predict,temperature,top_p,top_k,num_ctx, repeat_penalty, seed,num_gpu,stop])
    #存储model值
    model.change(save_model, inputs=[model], outputs=None, queue=True).then(model_run.params_update, None,None)
    #根据max_epoch选项更改cut_epoch的值
    max_epoch.change(cut_epoch, inputs=[max_epoch, cut_nepoch], outputs=[cut_nepoch])
    #保存cut_params到本地cache
    save_cut_param_button.click(save_cut_params, inputs=[max_epoch, cut_nepoch], outputs=None, queue=True).then(model_run.params_update, None, None)
    #保存params到本地cache
    save_model_loader_params_button.click(save_params, inputs=[num_predict,temperature,top_p,top_k,num_ctx, repeat_penalty, seed,num_gpu,stop], outputs=None, queue=True).then(model_run.params_update, None, None)
    #将各种参数保存到特定路径，并展示
    #根据refresh_params_dp选项更改保存参数按钮的标签
    refresh_params_dp.change(modify_label, inputs=[save_cus_params_button, refresh_params_dp], outputs=[save_cus_params_button], )

    #点击save_params_button 保存或更新参数文件
    save_cus_params_button.click(ccps, inputs=[saved_params_display,saved_params_display_copy, saved_params_display_tex, refresh_params_dp], outputs=[saved_params_display,saved_params_display_copy, saved_params_display_tex])
    #saved_params_display 更新参数文件
    saved_params_display.select(ccps, inputs=[saved_params_display,saved_params_display_copy, saved_params_display_tex, refresh_params_dp_copy], outputs=[saved_params_display, saved_params_display_copy, saved_params_display_tex])
    #当只有一个值的时候自动更新---解决一个值的时候选不上的bug----为完美解决
    #saved_params_display_copy 更新参数文件
    saved_params_display_copy.select(ccps, inputs=[saved_params_display,saved_params_display_copy, saved_params_display_tex, refresh_params_dp_copy], outputs=[saved_params_display, saved_params_display_copy, saved_params_display_tex])

    #根据saved_params_display修改params中各变量的展示
    saved_params_display.change(display_params_button, inputs=[saved_params_display, model, model_loader, 
        num_predict,temperature, top_p, top_k, num_ctx,repeat_penalty, seed, num_gpu,stop, max_epoch, cut_nepoch, promot_text], outputs=[ model, model_loader, 
        num_predict,temperature, top_p, top_k, num_ctx,repeat_penalty, seed, num_gpu,stop, max_epoch, cut_nepoch, promot_text], queue=True).then(model_run.params_update, None, None)
    #重命名参数组名称
    save_params_modify.submit(params_modify_textbox, inputs=[save_params_modify, saved_params_display, saved_params_display_copy, saved_params_display_tex], outputs=[saved_params_display,saved_params_display_copy, saved_params_display_tex])
    #删除参数组合名称
    save_params_delete.click(params_delete_button, inputs=[saved_params_display,saved_params_display_copy, saved_params_display_tex], outputs=[saved_params_display,saved_params_display_copy, saved_params_display_tex])
    
    #saved_params_display_copy与saved_params_display同步--不能使用change，否者为空的时候死循环
    saved_params_display_copy.select(dropdown_copy, inputs=[saved_params_display_copy, saved_params_display], outputs=[saved_params_display])
    saved_params_display.select(dropdown_copy, inputs=[saved_params_display, saved_params_display_copy], outputs=[saved_params_display_copy])

    #聊天记录存储以及输出
    #history:[[问，答]，[问，答]..], messages:[{'role':.., 'content':..},..]
    #save_chat_json:keys是时间，values=[history, messages]，点击save_chat,将聊天记录存储到本地，更新save_chat_json,
    save_chat_button.click(model_run.save_history_to_cache, None, None, queue=True).then(json_update, [save_chat_json], [save_chat_json])
    # 点击delete button，在chat json中删除指定的键值，将json值存储到本地
    delete_chat_record.click(delete_record, [chat_record, save_chat_json], [save_chat_json])
    # save_chat_json改变的时候更改chat_record的候选选项，更改chatbot相关的tts_have_read_len和chat_end_flag
    save_chat_json.change(update_record, [save_chat_json, chat_record], [chat_record])

    #chat_record更改时候更新save_chat_json值，更新tts_have_read_len并更新chatbot值
    chat_record_fn1=chat_record.change(json_update, [save_chat_json], [save_chat_json], queue=True).then(lambda xxx,yyy:refresh_chat_end_flag(yyy[xxx][0]), [chat_record, save_chat_json], None)#.then(lambda :gr.Number(value=0), None, chat_end_flag)
    chat_record_fn1.then(display_record_chat, [chat_record, save_chat_json, chatbot], [chatbot])
    #保存chat_record键值到本地，更新对话的messages记录
    chat_record.select(model_load_weight, chat_record, None, queue=True).then(model_run.load_messages, None, None)
    #当json仅有一个key时候chat_record blur时自动显示----不太行，待办

    #清空chatbot, 存入is_clear_chat值，以待model读取
    clear_button.click(clear_inform, inputs=[chatbot], outputs=[chatbot], queue=True).then(model_run.clear, None, None)

    #版本2的tts功能，与llm并行运行
    #1.开关按钮切换；2.开关tts api，3. TTS().run启用/关闭4.use_tts_on2:查找是否成功开启tts服务，传入tts_params.json文件的参数；use_tts_close2：关闭参数组合展示
    use_tts_on2.click(lambda:[gr.Button(visible=False), gr.Button(visible=True)], None, [use_tts_on2, use_tts_close2], queue=True).then(run_close_tts, [use_tts_on2, tts_bat_path_text], None).then(check_tts, None, use_tts_close2)

    use_tts_close2.click(lambda x:[gr.Button(visible=False, value="tts is launching"), gr.Button(visible=True)], use_tts_on2, [use_tts_close2, use_tts_on2], queue=True).then(display_hide_audio_params, [use_tts_close2], [tts_over_all_params]).then(run_close_tts, [use_tts_close2, tts_bat_path_text], None)

    # #点击按钮audio_params_submit将自定义tts参数保存到本地并传入tts_api,并更改自身的label值，再改回来label值
    audio_params_submit.click(lambda :gr.Button("params is saved and activate tts"), None,audio_params_submit, queue=True).then(params_overall_json_update, [audio_params_json, gpt_path,sovits_path,ref_audio_path, prompt_text, batch_size,prompt_lang,text_lang,audio_seed,text_split_method,speed_factor], [audio_params_json]).then(lambda tts_json:open(cache_path+'tts_params.json','w+').write(json.dumps(tts_json)), audio_params_json, None).then(lambda tts_json:tts_funcion.modify(tts_json['model_path'], tts_json['params']), audio_params_json, None).then(lambda :gr.Button("Save and Apply tts params"), None, audio_params_submit)

    #开启params栏目按钮
    show_tts_params.click(display_hide_audio_params, [gr.Button("click to start tts", visible=False)], [tts_over_all_params])
    
    hide_params_button.click(lambda x:gr.Column(visible=False), hide_params_button, tts_over_all_params)

    #ASR模块交互
    #点击按钮，更改label，检查asr_err是否初始化
    start_asr1 = start_asr.click(lambda :gr.Button("asr is ininting"), None, start_asr, queue=True,).then(asr_init, gr.Textbox(asr_api_path ,visible=False), None,).then(check_asr, None, [asr_pid, asr_url])
    #asr_launched，切换按钮显隐顺序，--加入asr_api地址，使用dropdown
    start_asr2 = start_asr1.then(lambda :[gr.Button("start asr", visible=False), gr.Button(visible=True)], None, [start_asr, end_asr])
    #发起asr_api请求的同时读取asr_err中的结果, 使用迭代运行一个计时器，计时器改变的时候进行读取get_asr_process，asr_request完成的时候取消计时器的迭代
    star_asr3 = start_asr2.then(asr_request, None, txt)
    tt_count_process = start_asr2.then(t_count, None, time_count)
    #每秒读取一次文件
    time_count.change(get_asr_process, None, asr_text_process)
    ##差一个提交文本， 假设txt submit成功后：先清空asr_text_process，继续请求语音转写
    chat_end_flag.change(lambda :gr.Textbox(None), None, asr_text_process).then(lambda x: asr_request() if x!=1 else None, chat_end_flag, txt)
    #关闭asr按钮---停止time_count的迭代，开启和关闭按钮显示隐藏切换，关闭asr_api的接口
    end_asr.click(lambda :[gr.Button(visible=False), gr.Button(visible=True)], None, [end_asr, start_asr],queue=True).then(lambda x,y:rc.close_pid_and_port(x,y), [asr_pid, asr_port], None).then(None,None, None, cancels=[tt_count_process],queue=True)

    # #model非None的时候运算
    # commit_btn.click(fn=add_text, inputs=[chatbot, txt], outputs=[chatbot, txt]).then(
    #     bot, chatbot, chatbot, api_name="bot_response"
    # ).then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

    #提交文本并生成对话。防止选择聊天记录的时候，tts_have_read_len里的旧值小于chatbot的原记录--(第一交互是冗余，只是害怕submit想不到的情况，所以暂时不改)
    #令tts_have_read_len是上一轮对话的回复长度
    txt_submit1 = txt.submit(lambda josn:refresh_chat_end_flag(josn['chatbot']), tts_read_process_json, None, queue=True, concurrency_limit=1)
    #更新chatbot和对话结束标识chat_end_flag
    txt_submit2 = txt_submit1.then(lambda x,y:[x+[(y, '')], gr.Textbox(value="", interactive=False)], [chatbot, txt], [chatbot, txt]).then(lambda:gr.Number(value=0), None, chat_end_flag,)
    #初始化tts_read_process_json和tts_have_read_len
    txt_submit3 = txt_submit2.then(refresh_chat_end_flag, gr.Chatbot([], visible=False), None, ).then(lambda :gr.Json({'chatbot':[["",""]], 'chat_end_flag':0}), None, tts_read_process_json)
    #生成回复
    txt_submit4 = txt_submit3.then(bot, chatbot, chatbot, api_name="bot_response").then(lambda:gr.Number(value=1), None, chat_end_flag).then(lambda x:gr.Textbox(interactive=True), txt, txt, queue=False)

    # cc = gr.Textbox(label="辅助测试")
    # chatbot.change(lambda x:x[-1][1], chatbot, cc)

    #当chatbot或者chat_end_flag变化时，将内容加入到json后，然后根据json内容统一单线程传入tts_text
    chatbot.change(tts_process_json, [chatbot,tts_read_process_json, gr.Textbox("chatbot",visible=False), ], tts_read_process_json, queue=True)
    chat_end_flag.change(tts_process_json, [chat_end_flag,tts_read_process_json, gr.Textbox("chat_end_flag",visible=False)], tts_read_process_json, queue=True)
    tts_read_process_json.change(tts_funcion.tts_text, tts_read_process_json, None, queue=True)

if __name__ == "__main__":

    demo.launch()