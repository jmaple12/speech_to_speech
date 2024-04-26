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

cache_path = os.path.abspath('../webui/cache/').replace('\\','/')+'/'
webui_path = os.path.abspath('../webui')
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

user_avatar = webui_path+"\\assets/user.png"
chatot_avatar = webui_path+"\\assets/chatbot.png"
user_avatar = user_avatar.replace('/','\\')
chatot_avatar = chatot_avatar.replace('/','\\')

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            #system_pormot
            promot_text = gr.Textbox(label='promot_text', value="", lines=1)
            promot_text.submit(save_promot_text, inputs=[promot_text], outputs=None)
            with gr.Row():
                #选择框架
                model_loader = found_model_loader()
                #选择模型
                model = found_model()
                model_loader.change(model_select_update, inputs=[model_loader, model], outputs=[model])
                #存储
                model_loader.change(save_model_loader, inputs=[model_loader], outputs=None)
                model.change(save_model, inputs=[model], outputs=None)

            # #选择最大对话轮次，超过时裁剪对话记录
            max_epoch, cut_nepoch = gr.Slider(visible=False),gr.Slider(visible=False)
            model_loader.change(gr_cut_params, inputs=[model_loader, max_epoch, cut_nepoch], outputs=[max_epoch, cut_nepoch])
            max_epoch.change(cut_epoch, inputs=[max_epoch, cut_nepoch], outputs=[cut_nepoch])
            gr.Interface(save_cut_params, inputs=[max_epoch, cut_nepoch], outputs='text')

            #选择模型参数---待办：参数用params代替
            num_predict = gr.Slider(visible=False)
            temperature = gr.Slider(visible=False)
            top_p = gr.Slider(visible=False)
            top_k = gr.Slider(visible=False)
            num_ctx = gr.Slider(visible=False)
            repeat_penalty = gr.Slider(visible=False)
            seed = gr.Number(visible=False)
            num_gpu = gr.Number(visible=False)
            stop = gr.Textbox(visible=False)
            model_loader.change(gr_model_params, inputs=[model_loader, num_predict,temperature,top_p,top_k,num_ctx, repeat_penalty, seed,num_gpu,stop], outputs=[num_predict,temperature,top_p,top_k,num_ctx, repeat_penalty, seed,num_gpu,stop])
            #将参数值存储cache/params.json
            gr.Interface(save_params, inputs=[num_predict,temperature,top_p,top_k,num_ctx, repeat_penalty, seed,num_gpu,stop], outputs='text')

            # with open('cache/model_val.txt','w+', encoding='utf-8') as file:
            #     model = file.read()
            # print('model:', model)
            model_run = Model(cache_path)
            bot = model_run.bot
        
        with gr.Column(scale=1, min_width=600):
            #定义聊天区域
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                bubble_full_width=False,
                #显示头像
                avatar_images=(user_avatar, (chatot_avatar)),
            )
            txt = gr.Textbox(
                scale=4,
                show_label=False,
                placeholder="Enter text and press enter to submit, shift+enter to newline",
                container=False,
                label="chat now",
            )

            #点击“chat”的反应
            commit_btn = gr.Button("chat",scale=2)
            #model非None的时候运算
            # if model !='None':
            commit_btn.click(fn=add_text, inputs=[chatbot, txt], outputs=[chatbot, txt]).then(
                bot, chatbot, chatbot, api_name="bot_response"
            ).then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

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


demo.queue()
if __name__ == "__main__":
    demo.launch()