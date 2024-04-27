# -*- coding: utf-8 -*-
import random
import gradio as gr
import os, re
import time
import json
from dotenv import load_dotenv, find_dotenv
import requests
import winsound
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
from asr import asr_request,check_asr, asr_init,get_asr_process,read_ast_out
#css
main_css = open('../webui/assets/style/main.css','r').read()
refresh_symbol = '🔄'
delete_symbol = '🗑️'
save_symbol = '💾'
interrupt_symbol = '⏹'


webui_path = os.path.abspath('../webui')
cache_path = os.path.abspath('../webui/cache/').replace('\\','/')+'/'
# asr_api_path = "E:\LargeModel\kaldi\sherpa_onnx_model\sherpa_onnx_speech_recognizier.bat"
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
    if use_tts == "点击开启TTS":
        return{tts_over_all_params:gr.Column(visible=True)}
    else:
        return{tts_over_all_params: gr.Column(visible=False)}

def run_close_tts(use_tts, tts_bat_path_upload):
    if use_tts == "点击开启TTS": 
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

def interrupt_rerun(use_tts_close2, ):
    if use_tts_close2 == "TTS开启中":
        tts_funcion.open_tts = False
    else:
        tts_funcion.open_tts = True


def tts_process_json(chatbot, chat_end_flag, tts_json, text):
    tts_json['chatbot'] = chatbot
    tts_json['chat_end_flag'] = chat_end_flag
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
            return(gr.Button("TTS启动完成，点击关闭"))
    print("api未开启")
    return


def loads_bat_path_text(tts_bat_path_text1, json_key='tts'):
    
    temp_val = json.loads(open(cache_path+'other_params_saved.json','r').read())
    temp_val.update({'%s_bat'%json_key:tts_bat_path_text1})
    if tts_bat_path_text1:
        print("更新%s_api文件位置为%s"%(json_key, tts_bat_path_text1))
    else:
        print("更新%s_api文件位置为默认值"%json_key)
    open(cache_path+'other_params_saved.json','w+').write(json.dumps(temp_val))

#时间间隔计数器
def t_count(time_sep=0.2, max_time=10**5):
    #默认最大计时间上限
    if not max_time:
        max_time=10**5
    if not time_sep:
        time_sep = 0.2
    for ttime in range(1,max_time):
        time.sleep(time_sep)
        yield(ttime)

def begin_next_epoch_asr():
    '''asr进行语音识别先加入一小段提示音表示转写开始'''
    winsound.Beep(500,500)
    return(asr_request())

def print_fun(y, x):
    print("temp_tts_flag:%s"%y,x)


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
head_name = ['tts','asr']
bat_path_text_cache = {head_nname:'' for head_nname in head_name}
with open(cache_path+'other_params_saved.json', 'r') as file:
    temp_val = file.read()
    if temp_val:
        temp_val =json.loads(temp_val)
        for head_nname in head_name:
            if head_nname+'_bat' in temp_val:
                if temp_val[head_nname+'_bat']:
                    bat_path_text_cache[head_nname] = temp_val[head_nname+'_bat']
                    print("读取%s_api的文件路径为%s"%(head_nname,temp_val[head_nname+'_bat']))
    for head_nname in head_name:
        if bat_path_text_cache[head_nname] =='':
            print("读取%s_api默认文件位置"%head_nname)

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

with gr.Blocks(css=main_css) as demo:
    # ----------------------------------------------------------#
    # --------------------Chat Params--------------------------#
    # ----------------------------------------------------------#
    with gr.Tab('LLM 参数'):
        with gr.Row():
            #保存参数
            with gr.Column(min_width=200):
                #展示保存的参数字典--展示键值
                saved_params_display = gr.Dropdown(label='保存的LLM参数组合', choices=myccp.display_params_name())
                #把键值存储到string里面
                saved_params_display_tex = gr.Textbox(visible=False)
                #字典值重命名
                save_params_modify = gr.Textbox(label='重命名上面的LLM参数组合名称')
            with gr.Column(min_width=300): 
                # 字典值删除    
                save_params_delete = gr.Button('点击删除选中的参数组合')
                with gr.Row():
                    #保存参数按钮save_params_button--值不能随便改
                    save_cus_params_button = gr.Button('点击保存选中的参数')
                    #对保存参数按钮增加几个选项---1。保存参数+覆盖源文件，2。保存参数+不覆盖源文件，3。刷新参数
                    refresh_params_dp = gr.Radio(label='新增或者覆盖参数组合', choices=['add', 'cover'], value='add')
            
            restart_ollama = gr.Button('重启Ollama APP')

        with gr.Row():
            #system_pormot
            with gr.Column(min_width=400):
                promot_text = gr.Textbox(label='输入Promot 文本，按Enter保存', value="", lines=1)
                #提交动画：label:显示文本已经提交
                promot_text.submit(lambda:gr.Textbox(label="文本已提交"), None, promot_text, queue=True).then(lambda :t_count(2,2),None, None).then(lambda :gr.Textbox(label="输入Promot 文本，按Enter保存"), None, promot_text)
                with gr.Row():
                    #选择框架
                    model_loader = found_model_loader()
                    #选择模型
                    model = found_model()
                #save_messages_cut_params button
                save_cut_param_button = gr.Button('点击保存聊天记录截断参数', visible=False)
                #点击动画：
                save_cut_param_button.click(lambda :gr.Button('截断参数已保存'), None,save_cut_param_button, queue=True).then(lambda :gr.Button('点击保存聊天记录截断参数'),None,save_cut_param_button,)
                # #选择最大对话轮次，超过时裁剪对话记录
                max_epoch, cut_nepoch = gr.Slider(visible=False),gr.Slider(visible=False)

            with gr.Column(min_width=100, scale=1):
                #将参数值存储cache/params.json
                save_model_loader_params_button = gr.Button('点击保存model_loader参数', visible=False)
                #点击动画：
                save_model_loader_params_button.click(lambda:gr.Button('model_loader参数已保存'), None,save_model_loader_params_button, queue=True).then(lambda :gr.Button('点击保存model_loader参数'), None, save_model_loader_params_button,)
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
    with gr.Tab('聊天'):
        with gr.Row():
            with gr.Column(min_width=200):
                #聊天记录保存
                #保存的聊天记录存储到save_chat_json(隐藏)
                save_chat_json = gr.Json({}, visible=False, value=model_run.load_history_json_file())
                with gr.Row():
                    #下拉框的额默认值为
                    chat_record = gr.Dropdown(label='聊天记录',choices=sorted(model_run.load_history_json_file().keys()),)
                    #刷新聊天记录按钮
                    # refresh_chat_record = gr.Button('refresh chat record')
                    #删除聊天记录按钮
                    delete_chat_record = gr.Button('删除这条记录')
                
                #tts
                #tts_bat_path
                ##gradio将文件放到temp中的一个路径，地址也是这个地址，所以不能cd..，因此不能用Uploadbutton
                tts_bat_path_text = gr.Textbox(label="输入TTS API(.bat)文件的地址",value=bat_path_text_cache['tts'].strip('\'').strip('"') if bat_path_text_cache['tts'] else None,)
                tts_bat_path_text.submit(lambda xxx:loads_bat_path_text(xxx, 'tts'), tts_bat_path_text, None)
                with gr.Row():                  
                    #版本2
                    use_tts_on2 = gr.Button("点击开启TTS",)
                    use_tts_close2 = gr.Button("TTS开启中", visible=False) 
                    #是否开启tts服务。0表示未开启，1表示开启
                    start_tts_service = gr.Number(0, label="是否开启TTS服务", visible=False)
                #对话时，每回复几句话时将生成的文本传给tts
                tts_endure = gr.Slider(minimum=1, maximum=20, step=1, value=2, label="tts edure sentence num")
                tts_endure.release(tts_funcion.modify_endure_sentence, tts_endure, None)

                #当chatbot或者chat_end_flag变化时，将内容加入到json后，然后根据json内容统一单线程传入tts_text
                tts_read_process_json = gr.Json({'chatbot':[['','']], 'chat_end_flag':0}, visible=False)
                #点击展示tts的可调节参数
                show_tts_params = gr.Button("点击显示TTS参数")
                #tts,每一轮 text to speech转写完成的标志
                tts_finish_flag =gr.Number(0, label="TTS转译完成标志", visible=False)

                
            with gr.Column(min_width=400, scale=6, visible=True):
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
                        scale=30,
                        show_label=False,
                        placeholder="Enter提交，Shift+Enter换行",
                        container=False,
                        label="chat now",
                    )
                    txt_submit = gr.Textbox(
                        label="submit txt",
                        value="", 
                        visible=False, 
                        scale=30,
                        container=False,
                        placeholder="submit txt",
                        show_label=False
                    )
                    #是否提交txt_submit的标识，当txt_flag为1的时候asr进行录音·
                    txt_flag = gr.Number(0, visible=False)
                    ##清屏+保存聊天记录=开启新聊天
                    #chatbot清屏按钮
                    clear_button = gr.Button(refresh_symbol, scale=0, elem_classes='small-button')

                    #chatbot保存聊天记录按钮
                    save_chat_button = gr.Button(save_symbol, scale=0, elem_classes='small-button')
                    #中断chatbot的按钮
                    stop_chat_button = gr.Button(interrupt_symbol, scale=0, elem_classes='small-button')
                    
                # #点击“chat”的反应
                # commit_btn = gr.Button("chat",scale=1)

            with gr.Column(min_width=200):
                #展示保存的参数字典--展示键值
                saved_params_display_copy = gr.Dropdown(label='保存的LLM参数组合', choices=myccp.display_params_name())
                #固定值的按钮---代替字符串使用
                refresh_params_dp_copy = gr.Textbox(value='refresh', visible=False)

                #-----------------ASR----------------------------------#
                #计时器
                time_count = gr.Number(0, visible=False, label='计时器')
                ##开启asr按钮---打开
                with gr.Row():
                    #是否开启asr服务
                    start_asr_flag = gr.Number(value=0, label="是否开启ASR服务", visible=False)
                    asr_text_process = gr.TextArea(label="ASR进度转写记录", show_label=False, placeholder="ASR进度转写记录", scale=30, container=False,)
                with gr.Row(visible=False):
                    asr_pid = gr.Textbox(visible=True, label="asr_pid")
                    asr_url = gr.Textbox(visible=True, label="asr_url")
                    asr_port = gr.Textbox(visible=True, label="asr_port")
                    asr_url.change(lambda x:x.split(':')[-1] if ':' in x else None, asr_url, asr_port)
                asr_show_len = gr.Number(0, label="the asr_text_process has shown length for asr", visible=False)
                #asr_api地址
                asr_api_path = gr.Textbox(label='ASR API(.bat)文件位置', value=bat_path_text_cache['asr'].strip('\'').strip('"') if bat_path_text_cache['asr'] else None,)
                asr_api_path.submit(lambda xxx:loads_bat_path_text(xxx, 'asr'), asr_api_path, None)
                with gr.Row():
                    #star_asr的值不能轻易改
                    start_asr = gr.Button("启动ASR",)
                    end_asr = gr.Button("ASR已开启，点击关闭", visible=False)
                    #当语音对话结束时，开启新一轮对话的按钮
                    new_asr_request = gr.Button("录入语音并发送")

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
                gpt_path = gr.Textbox(label='GPT模型的文件路径(.ckpt)', visible=True, value=tts_params_default['model_path']['gpt_path'])
                sovits_path = gr.Textbox(label='Sovits模型的文件路径(.pth)', visible=True, value=tts_params_default['model_path']['sovits_path'])
                ref_audio_path = gr.Textbox(label='参考音频的文件路径(.wav)', visible=True, value=tts_params_default['params']['ref_audio_path'])

            with gr.Row():
                prompt_text = gr.Textbox(label='输入参考音频的字幕', value=tts_params_default['params']['prompt_text'])
                audio_seed = gr.Number(label='seed', value=tts_params_default['params']['seed'])
            with gr.Row():    
                batch_size = gr.Slider(minimum=1, maximum=20, step=1, value=tts_params_default['params']['batch_size'], label='batch_size')
                #语速
                speed_factor = gr.Slider(minimum=0.1, maximum=3, step=0.1, value=tts_params_default['params']['speed_factor'], label='语速')

            with gr.Row():
                #切分方式
                text_split_method_choices = ["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切", "根据句末结束符(中英文。？！等)切"]
                text_split_method_show = gr.Dropdown(text_split_method_choices, label='TTS文本切分方式', value= text_split_method_choices[int(tts_params_default['params']['text_split_method'][3:])])
                text_split_method = gr.Dropdown(['cut'+str(i) for i in range(7)],label='prompt_text', value=tts_params_default['params']['text_split_method'], visible=False)            
                text_split_method_show.select(lambda zh:gr.Dropdown(value="cut"+str([item[0] for item in enumerate(["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切", "根据句末结束符(中英文。？！等)切"]) if item[1]==zh][0])), text_split_method_show, text_split_method)
                #promot_word language
                prompt_lang = gr.Dropdown(['zh','en','ja'],label='参考音频的语言', value=tts_params_default['params']['prompt_lang'])
                #文本语言
                text_lang = gr.Dropdown(['zh','en','ja'], label='文本语言', value=tts_params_default['params']['text_lang'])
            
            #保存参数的按钮
            with gr.Row():
                audio_params_submit = gr.Button('保存并应用TTS参数')
                audio_params_json = gr.Json(value={'model_path':{}, 'params':{}}, visible=False)
                hide_params_button = gr.Button("点击隐藏TTS参数")
    
    #点击重启ollama
    restart_ollama.click(lambda :gr.Button("重启中...请等待3秒"), None, restart_ollama, queue=True).then(rc.clear_rerun_ollama, None, None).then(lambda :gr.Button("重启Ollama APP"), None, restart_ollama, )
    #存储promot_text
    promot_text.submit(save_promot_text, inputs=[promot_text], outputs=None, queue=True).then(model_run.add_promot, None, None,)
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
    model.change(save_model, inputs=[model], outputs=None, queue=True).then(model_run.params_update, None,None,)
    #根据max_epoch选项更改cut_epoch的值
    max_epoch.change(cut_epoch, inputs=[max_epoch, cut_nepoch], outputs=[cut_nepoch])
    #保存cut_params到本地cache
    save_cut_param_button.click(save_cut_params, inputs=[max_epoch, cut_nepoch], outputs=None, queue=True).then(model_run.params_update, None, None,)
    #保存params到本地cache
    save_model_loader_params_button.click(save_params, inputs=[num_predict,temperature,top_p,top_k,num_ctx, repeat_penalty, seed,num_gpu,stop], outputs=None, queue=True).then(model_run.params_update, None, None,)
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
        num_predict,temperature, top_p, top_k, num_ctx,repeat_penalty, seed, num_gpu,stop, max_epoch, cut_nepoch, promot_text], queue=True).then(model_run.params_update, None, None,)
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
    chat_record_fn1=chat_record.change(json_update, [save_chat_json], [save_chat_json], queue=True).then(lambda xxx,yyy:refresh_chat_end_flag(yyy[xxx][0]), [chat_record, save_chat_json], None,)
    chat_record_fn1.then(display_record_chat, [chat_record, save_chat_json, chatbot], [chatbot],)
    #保存chat_record键值到本地，更新对话的messages记录
    chat_record.select(model_load_weight, chat_record, None, queue=True).then(model_run.load_messages, None, None,)
    #当json仅有一个key时候chat_record blur时自动显示----不太行，待办

    #清空chatbot, 存入is_clear_chat值，以待model读取
    clear_button.click(clear_inform, inputs=[chatbot], outputs=[chatbot], queue=True).then(model_run.clear, None, None,)

    #版本2的tts功能，与llm并行运行
    #1.开关按钮切换；2.开关tts api，3. TTS().run启用/关闭；4.use_tts_on2:查找是否成功开启tts服务，传入tts_params.json文件的参数；use_tts_close2：关闭参数组合展示||读取chatbot内容的代码在txt_submit那里；5.更改tts是否开启的标识start_tts_service
    use_tts_on2.click(lambda:[gr.Button(visible=False), gr.Button(visible=True)], None, [use_tts_on2, use_tts_close2], queue=True).then(run_close_tts, [use_tts_on2, tts_bat_path_text], None).then(check_tts, None, use_tts_close2).then(lambda :gr.Number(value=1), None, start_tts_service, )

    use_tts_close2.click(lambda :[gr.Button(visible=False, value="TTS开启中"), gr.Button(visible=True)], None, [use_tts_close2, use_tts_on2], queue=True).then(display_hide_audio_params, [use_tts_close2], [tts_over_all_params]).then(run_close_tts, [use_tts_close2, tts_bat_path_text], None).then(lambda :gr.Number(value=0), None, start_tts_service, )

    # #点击按钮audio_params_submit将自定义tts参数保存到本地并传入tts_api,并更改自身的label值，再改回来label值
    audio_params_submit.click(lambda :gr.Button("TTS参数已保存并激活"), None,audio_params_submit, queue=True).then(params_overall_json_update, [audio_params_json, gpt_path,sovits_path,ref_audio_path, prompt_text, batch_size,prompt_lang,text_lang,audio_seed,text_split_method,speed_factor], [audio_params_json]).then(lambda tts_json:open(cache_path+'tts_params.json','w+').write(json.dumps(tts_json)), audio_params_json, None).then(lambda tts_json:tts_funcion.modify(tts_json['model_path'], tts_json['params']), audio_params_json, None).then(lambda :gr.Button("保存并应用TTS参数"), None, audio_params_submit,)

    #开启params栏目按钮
    show_tts_params.click(display_hide_audio_params, [gr.Button("点击开启TTS", visible=False)], [tts_over_all_params])
    
    hide_params_button.click(lambda :gr.Column(visible=False), None, tts_over_all_params)

    #ASR模块交互
    #点击按钮，更改label，更改txt和txt_submit的显示隐藏,归零asr_show_len已读asr_err长度，关闭可能的asr线程, 初始化asr
    start_asr0 = start_asr.click(lambda :gr.Button("ASR启动中"), None, start_asr, queue=True,).then(lambda :[gr.Textbox(visible=False), gr.Textbox(visible=True)], None, [txt, txt_submit]).then(lambda :gr.Number(0), None, asr_show_len).then(read_ast_out, None, [asr_pid, asr_port]).then(rc.close_pid_and_port, [asr_pid, asr_port], None).then(asr_init,asr_api_path, None,)
    # 检查asr_err是否初始化
    start_asr1 = start_asr0.then(check_asr, None, [asr_pid, asr_url])
    
    #asr_launched，切换按钮显隐顺序，更新start_asr_flag的值
    start_asr2 = start_asr1.then(lambda :[gr.Button("启动ASR", visible=False), gr.Button(visible=True)], None, [start_asr, end_asr]).then(lambda:gr.Number(value=1), None, start_asr_flag)
    
    #发起asr_api请求的同时读取asr_err中的结果, 使用迭代运行一个计时器，计时器改变的时候进行读取get_asr_process，asr_request完成的时候取消计时器的迭代
    tt_count_process = start_asr2.then(t_count, [gr.Number(value=0.5, visible=False), gr.Textbox(value=None, visible=False)], time_count)
    star_asr3 = start_asr2.then(begin_next_epoch_asr, None, txt_submit)

    #每秒读取一次文件
    time_count.change(get_asr_process, asr_show_len, [asr_text_process,asr_show_len], show_progress='hidden')
    
    #一轮tts转写结束及tts_finish_flag变为1后若asr服务开启，请求语音转写--使用winsound.Beep(500,500)作为接受语音进行转写的提示音
    tts_finish_flag.change(lambda xxx,yyy: begin_next_epoch_asr() if (xxx==1)&(yyy==1) else "", [tts_finish_flag, start_asr_flag], txt_submit)

    #txt_submit提交成功后：先清空asr_text_process，当tts服务未开启时，asr服务开启时，请求语音转写
    chat_end_flag.change(lambda :gr.TextArea(None), None, asr_text_process).then(lambda xxx,yyy,zzz: begin_next_epoch_asr() if (xxx==1)&(yyy==0)&(zzz==1) else "", [chat_end_flag,start_tts_service, start_asr_flag], txt_submit)

    #手动开启新的一轮语音对话，即发起asr_request
    new_asr_request.click(begin_next_epoch_asr, None, txt_submit)

    #关闭asr按钮---停止time_count的迭代，开启和关闭按钮显示隐藏切换，txt和txt_submit的显示隐藏切换，关闭asr_api的接口，将start_asr_flag归0
    end_asr.click(lambda :[gr.Button(visible=False), gr.Button(visible=True)], None, [end_asr, start_asr],queue=True, cancels=[tt_count_process]).then(lambda x,y:rc.close_pid_and_port(x,y), [asr_pid, asr_port], None).then(lambda :[gr.Textbox(visible=False), gr.Textbox(visible=True)], None, [txt_submit,txt]).then(lambda:gr.Number(value=0), None, start_asr_flag, )

    #提交文本并生成对话。防止选择聊天记录的时候，tts_have_read_len里的旧值小于chatbot的原记录--(第一交互是冗余，只是害怕submit想不到的情况，所以暂时不改)。令tts_have_read_len是上一轮对话的回复长度
    txt.submit(lambda x:x, txt, txt_submit,queue=True).then(lambda :gr.Textbox(value=None, interactive=False), None, txt, )

    #当txt_submit变为None或者''时不提交给chatbot，否则提交，使用txt_flag作为是否提交的标识，当txt_flag变动时，提交。
    txt_submit.change(lambda xxx,yyy:1-yyy if xxx else yyy, [txt_submit,txt_flag], txt_flag)
    txt_submit1 = txt_flag.change(lambda josn:refresh_chat_end_flag(josn['chatbot']), tts_read_process_json, None, queue=True, concurrency_limit=1)

    #更新chatbot内容，添加[txt_submit,'']；把对话结束标识chat_end_flag以及tts完成标识tts_finish_flag归零
    txt_submit2 = txt_submit1.then(lambda x,y:[x+[(y, '')], gr.Textbox(value="", interactive=False)], [chatbot, txt_submit], [chatbot, txt_submit],).then(lambda:gr.Number(value=0), None, chat_end_flag,).then(lambda:gr.Number(value=0), None, tts_finish_flag)

    #初始化tts_read_process_json和tts_have_read_len
    txt_submit3 = txt_submit2.then(lambda :gr.Json({'chatbot':[["",""]], 'chat_end_flag':0}), None, tts_read_process_json).then(refresh_chat_end_flag, gr.Chatbot([], visible=False), None, )
    #生成回复
    txt_submit4 = txt_submit3.then(bot, chatbot, chatbot, api_name="bot_response")
    txt_submit5 = txt_submit4.then(lambda:gr.Number(value=1), None, chat_end_flag).then(lambda :[gr.Textbox(interactive=True),gr.Textbox(interactive=True)], None, [txt_submit, txt], queue=False)

    #当chatbot或者chat_end_flag变化时，将内容加入到tts_read_process_json后，然后根据tts_read_process_json内容统一单线程传入tts.tts_text
    chatbot.change(tts_process_json, [chatbot, chat_end_flag,tts_read_process_json, gr.Textbox("chatbot",visible=False)], tts_read_process_json)
    #当chat_end_flag为0的时候不改变tts_read_process_json，因为在txt_submit中将chat_end_flag改为0时，tts_read_process_json还没有来得及因为chatbot的改变而改变，导致有时候tts会因为chat_end_flag的改变读取上一轮的结果；gr.json作为outputs时，即使值没有变，json.change也会生效，因此会将上一轮的文本传入tts_text:删除当chat_end_flag变为0时tts_read_process_json不变的代码
    chat_end_flag.change(lambda zzz,xxx,yyy: tts_process_json(zzz,xxx, yyy, 'chat_end_flag'), [chatbot, chat_end_flag,tts_read_process_json], tts_read_process_json)

    #tts_read_process_json内容变更时，将包含chatbot和chat_end_flag的tts_read_process_json输入到tts.tts_text中进行text to speech，当chat_end_flag变为1的时候，tts可能还在运行中，无法响应tts_read_process_json.change，因此加上temp_tts_flag；
    #temp_tts_flag是判定一段tts是否结束，如果结束，并且chat_end_flag=1，则再进行一次tts_text，并将结果返回给tts_finish_flag，这样可以确保所有生成的文本都能被转成语音。
    ##tts_read_process_json这种change是当内容更新的时候就会更改，内容不一定变了||待办：有时候的语音回复，在refresh和bot之间会出现一个调用tts_text，调用的json是上一轮的结果...保险3：temp_tts_flag改变时候将have_read_length与chatbot回复长度对比
    temp_tts_flag = gr.Number(value=0, visible=False)
    tts_read_process_json1 = tts_read_process_json.change(lambda :[gr.Number(value=0),gr.Number(value=0)], None, [tts_finish_flag,temp_tts_flag], queue=True)
    tts_read_process_json2 = tts_read_process_json1.then(tts_funcion.tts_text, tts_read_process_json, temp_tts_flag)
    #为tts转换语音上一层保险
    temp_tts_flag1 = temp_tts_flag.change(lambda xxx,yyy,zzz:tts_funcion.tts_text(yyy) if (zzz==1)&(xxx==1) else 0, [temp_tts_flag, tts_read_process_json, chat_end_flag], tts_finish_flag)

    #中断回复, cancels只会中断action中的head
    stop_chat_button.click(lambda :tts_funcion.tts_text({'chatbot':[], 'chat_end_flag':0}), None, tts_finish_flag, cancels=[txt_submit4, tts_read_process_json2, temp_tts_flag1], queue=True).then(interrupt_rerun, use_tts_close2, None)

if __name__ == "__main__":

    demo.launch(
        inbrowser=True,
        # quiet=True,
    )