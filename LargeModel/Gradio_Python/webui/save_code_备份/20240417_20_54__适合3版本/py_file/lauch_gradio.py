import random
import gradio as gr
import os, re
import time
import json
from dotenv import load_dotenv, find_dotenv
import requests
from model import Model
from load_model import found_model, found_model_loader, model_select_update, save_model,save_model_loader,save_promot_text
from model_params import gr_cut_params,gr_model_params,cut_epoch,save_params,save_cut_params
from save_params import create_cus_params,display_params_button,ccps,params_modify_textbox, params_delete_button,modify_label
from chat_function import clear_inform, json_default, json_update, display_record_chat,model_load_weight, delete_record , update_record
from tts import params_overall_json_update
from py_run_cmd import run_cmd

webui_path = os.path.abspath('../webui')
cache_path = os.path.abspath('../webui/cache/').replace('\\','/')+'/'

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
    copy = gr.Dropdown.update(value=origin)
    return(copy)

def curr_time():
    return(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

def display_hide_audio_params(use_tts):
    if use_tts:
        #等待tts开启
        time.sleep(3)
        return{tts_over_all_params:gr.Column(visible=True),
        prompt_lang:gr.Dropdown.update(value='zh'),
        text_lang:gr.Dropdown.update(value='zh')}
    else:
        return{tts_over_all_params: gr.Column(visible=False),
        prompt_lang : gr.Dropdown.update(value=None),
        text_lang : gr.Dropdown.update(value=None)}

def run_close_tts(use_tts, tts_bat_path_upload):
    print('use_tts:', use_tts)
    if use_tts:
        if tts_bat_path_upload:
            rc.gpt_tts_run(tts_bat_path_upload.name)
        else:
            rc.gpt_tts_run()
    else:
        rc.gpt_tts_close()


user_avatar = webui_path+"\\assets/user.png"
chatot_avatar = webui_path+"\\assets/chatbot.png"
user_avatar = user_avatar.replace('/','\\')
chatot_avatar = chatot_avatar.replace('/','\\')

#保存参数组合的管理系统：myccp-
myccp = create_cus_params()
#模型：model_run
model_run = Model(cache_path)
if model_run.use_tts:
    model_run.add_tts()
bot = model_run.bot
#python模拟cmd系统
rc = run_cmd()

with gr.Blocks() as demo:
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
            with gr.Column(min_width=200): 
                #保存参数按钮save_params_button
                save_cus_params_button = gr.Button('click to add current all params to particular path', scale=2)
                #对保存参数按钮增加几个选项---1。保存参数+覆盖源文件，2。保存参数+不覆盖源文件，3。刷新参数
                refresh_params_dp = gr.Radio(label='save params with add or cover', choices=['add', 'cover'], value='add')
            # 字典值删除    
            save_params_delete = gr.Button('click to delete key name')

        with gr.Row():
            #system_pormot
            with gr.Column(min_width=400):
                promot_text = gr.Textbox(label='promot_text, enter to save', value="", lines=1)
                
                with gr.Row():
                    #选择框架
                    model_loader = found_model_loader()
                    #选择模型
                    model = found_model()
                #save_messages_cut_params button
                save_cut_param_button = gr.Button('click to save cut_params', visible=False)
                # #选择最大对话轮次，超过时裁剪对话记录
                max_epoch, cut_nepoch = gr.Slider(visible=False),gr.Slider(visible=False)

            with gr.Column(min_width=100, scale=1):
                #将参数值存储cache/params.json
                save_model_loader_params_button = gr.Button('click to save model_loader params', visible=False)
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
                with gr.Row():
                    #tts_bat_path
                    tts_bat_path_upload = gr.File(label="choose gpt_tts bat file")
                    #tts开关按钮
                    use_tts = gr.Checkbox(value=False, label='launch tts use default params', scale=4)

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
                with gr.Row():
                    txt = gr.Textbox(
                        scale=20,
                        show_label=False,
                        placeholder="Enter text and press enter to submit, shift+enter to newline",
                        container=False,
                        label="chat now",
                    )
                    #清屏按钮
                    clear_button = gr.Button('clear_chat', size='sm')
                    #保存聊天记录按钮
                    save_chat_button = gr.Button('save_chat', size='sm')
                    
                # #点击“chat”的反应
                # commit_btn = gr.Button("chat",scale=1)
            
            #tts_参数
            with gr.Column(visible=False) as tts_over_all_params:
                with gr.Row():
                    with gr.Column():
                        gpt_path = gr.Textbox(label='upload gpt model(.ckpt)')
                        gpt_path_upload = gr.UploadButton(size='sm')
                        gpt_path_upload.upload(lambda x,y:gr.Textbox.update(y.name), [gpt_path, gpt_path_upload], [gpt_path])
                    with gr.Column():
                        sovits_path = gr.Textbox(label='upload gpt model(.ckpt)')
                        sovits_path_upload = gr.UploadButton(size='sm')
                        sovits_path_upload.upload(lambda x,y:gr.Textbox.update(y.name), [sovits_path, sovits_path_upload], [sovits_path])
                    with gr.Column():
                        ref_audio_path = gr.Textbox(label='upload ref audio')
                        ref_audio_path_upload = gr.UploadButton(size='sm')
                        ref_audio_path_upload.upload(lambda x,y:gr.Textbox.update(y.name), [ref_audio_path, ref_audio_path_upload], [ref_audio_path])
                with gr.Row():
                    prompt_text = gr.Textbox(label='input refer audio words')
                    seed = gr.Number(value=42, label='seed')
                with gr.Row():    
                    batch_size = gr.Slider(minimum=1, maximum=20, step=1, value=8, label='batch_size')
                    speed_factor = gr.Slider(minimum=0.1, maximum=3, step=0.1, value=1, label='speed_factor')

                with gr.Row():
                    text_split_method = gr.Dropdown(['cut'+str(i) for i in range(6)],label='prompt_text', value='cut5')
                    prompt_lang = gr.Dropdown(['zh','en','ja'],label='prompt_text')
                    text_lang = gr.Dropdown(['zh','en','ja'], label='text_lang')
                
                #保存参数的按钮
                audio_params_submit = gr.Button('Submit to save audio params and activate tts')
                audio_params_json = gr.Json(value={'model_path':{}, 'params':{}}, visible=False)
                hide_params_button = gr.Button("click to hide tts params")

    #存储promot_text
    promot_text.submit(save_promot_text, inputs=[promot_text], outputs=None, queue=True).then(model_run.add_promot, None, None)
    #存储model_loader值
    #根据model_loader展示model值
    #根据model_loader选项打开cut_params
    #根据model_loader选项展示params中各变量
    model_loader.change(save_model_loader, inputs=[model_loader], outputs=None, queue=True).then(model_run.params_update, None,None).then(model_select_update, inputs=[model_loader, model], outputs=[model]).then(gr_cut_params, inputs=[model_loader, max_epoch, cut_nepoch,save_cut_param_button], outputs=[max_epoch, cut_nepoch, save_cut_param_button]).then(gr_model_params, inputs=[model_loader,save_model_loader_params_button, num_predict,temperature,top_p,top_k,num_ctx, repeat_penalty, seed,num_gpu,stop], outputs=[save_model_loader_params_button, num_predict,temperature,top_p,top_k,num_ctx, repeat_penalty, seed,num_gpu,stop])
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
    save_params_modify.submit(params_modify_textbox, inputs=[save_params_modify, saved_params_display,saved_params_display_copy, saved_params_display_tex], outputs=[saved_params_display,saved_params_display_copy, saved_params_display_tex])
    #删除参数组合名称
    save_params_delete.click(params_delete_button, inputs=[saved_params_display,saved_params_display_copy, saved_params_display_tex], outputs=[saved_params_display,saved_params_display_copy, saved_params_display_tex])
    
    #saved_params_display_copy与saved_params_display同步
    saved_params_display_copy.change(dropdown_copy, inputs=[saved_params_display_copy, saved_params_display], outputs=[saved_params_display])
    saved_params_display.change(dropdown_copy, inputs=[saved_params_display, saved_params_display_copy], outputs=[saved_params_display_copy])

    #聊天记录存储以及输出
    #history:[[问，答]，[问，答]..], messages:[{'role':.., 'content':..},..]
    save_chat_button.click(model_run.save_history_to_cache, None, None, queue=True).then(json_update, [save_chat_json], [save_chat_json])
    chat_record.select(json_update, [save_chat_json], [save_chat_json])
    #save_chat_json:keys是时间，values=[history, messages]，点击save_chat,更新save_chat_json，chat_recrd
    delete_chat_record.click(delete_record, [chat_record, save_chat_json], [save_chat_json])
    save_chat_json.change(update_record, [save_chat_json, chat_record], [chat_record])
    chat_record.select(display_record_chat, [chat_record, save_chat_json, chatbot], [chatbot])
    #更新对话的messages记录
    chat_record.select(model_load_weight, chat_record, None, queue=True).then(model_run.load_messages, None, None)
    #当json仅有一个key时候chat_record blur时自动显示----不太行，待办

    #清空chatbot, 存入is_clear_chat值，以待model读取
    clear_button.click(clear_inform, inputs=[chatbot], outputs=[chatbot], queue=True).then(model_run.clear, None, None)

    #确认开启tts，取消勾选关闭tts
    use_tts.change(run_close_tts, [use_tts, tts_bat_path_upload], None, queue=True).then(lambda use_tts:model_run.add_tts() if use_tts else model_run.dismiss_tts(), use_tts, None)
    #显示隐藏不需要queue
    use_tts.change(display_hide_audio_params, [use_tts], [tts_over_all_params, prompt_lang, text_lang])
    #点击按钮audio_params_submit发起自定义参数的tts
    audio_params_submit.click(params_overall_json_update, [audio_params_json, gpt_path,sovits_path,ref_audio_path, prompt_text, batch_size,prompt_lang,text_lang,seed,text_split_method,speed_factor], [audio_params_json], queue=True).then(lambda json:model_run.add_tts(json['model_path'], json['params']), audio_params_json, None)
    hide_params_button.click(lambda x:gr.Column.update(visible=False), hide_params_button, tts_over_all_params)

    # #model非None的时候运算
    # commit_btn.click(fn=add_text, inputs=[chatbot, txt], outputs=[chatbot, txt]).then(
    #     bot, chatbot, chatbot, api_name="bot_response"
    # ).then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

    #文本提交的反应
    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=True).then(
        bot, chatbot, chatbot, api_name="bot_response"
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    # with gr.Column(scale=2, min_width=600):
    #     prompt_input = gr.Textbox(placeholder="Enter prompt for image generation", label="Image Prompt")
    #     navigator_prompt_input = gr.Textbox(placeholder="Enter navigator prompt", label="Navigator Prompt")

    #     gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery", object_fit="contain",
    #                          height="auto")

    #     model_select = gr.Dropdown(choices=["Model 1", "Model 2", "Model 3"], label="Choose a Model")
    #     btn = gr.Button("Generate images")
    #     btn.click(
    #         generate_images_with_prompts,
    #         [prompt_input, navigator_prompt_input, model_select],
    #         gallery
    #     )

#queue并行的数目---因为tts后端开启，所以必须>1
demo.queue(concurrency_count=4)
if __name__ == "__main__":

    demo.launch()