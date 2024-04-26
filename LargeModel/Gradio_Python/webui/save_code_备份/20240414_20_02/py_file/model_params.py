import gradio as gr
import json
def gr_cut_params(model, max_epoch, cut_nepoch):
    #生成cut_params
    if model=="ollama":
        #cut_nepoch随着max_epoch改变
        max_epoch = gr.Slider(minimum=1,maximum=100,step=1,label='max_epoch',value=15,interactive=True, visible=True)
        cut_nepoch = gr.Slider(minimum=0,maximum=15,step=1,label='cut_nepoch',value=5,interactive=True, visible=True)
    elif model =="None":
        max_epoch = gr.Slider(visible=False)
        cut_nepoch = gr.Slider(visible=False)
    return(max_epoch, cut_nepoch)

def cut_epoch(max_epoch, cut_nepoch):
    #根据max_epoch生成cut_nepoch
    cut_nepoch = gr.Slider(minimum=0,maximum=max_epoch,step=1,label='cut_nepoch',value=max(max_epoch//3,1),interactive=True, visible=True)
    return(cut_nepoch)

def save_cut_params(max_epoch, cut_nepoch):
    cut_params = {
        'max_epoch':max_epoch,
        'cut_nepoch':cut_nepoch
    }
    with open('.\cache\cut_params.json','w+') as file:
        file.write(json.dumps(cut_params))
    return('params has been saved')


def gr_model_params(model,
                num_predict,
                temperature,
                top_p,
                top_k,
                num_ctx,
                repeat_penalty,
                seed,
                num_gpu,
                stop
                ):
    if model =='None':
        num_predict = gr.Slider(visible=False)
        temperature = gr.Slider(visible=False)
        top_p = gr.Slider(visible=False)
        top_k = gr.Slider(visible=False)
        num_ctx = gr.Slider(visible=False)
        repeat_penalty = gr.Slider(visible=False)
        seed = gr.Number(visible=False)
        num_gpu = gr.Number(visible=False)
        stop = gr.Textbox(visible=False)
    elif model =="ollama":
        ###num_predict, num_ctx都是2的幂次, num_gpu要判断是否为-1，stop要按\n拆分--model.py---done
        num_predict = gr.Slider(minimum=5,maximum=14,step=1,label='2^num_predict',value=7,interactive=True, visible=True)
        temperature = gr.Slider(minimum=0.1,maximum=1,step=0.1,label='temperature',value=0.8,interactive=True, visible=True)
        top_p = gr.Slider(minimum=0.1,maximum=1,step=0.1,label='top_p',value=0.9,interactive=True, visible=True)
        top_k = gr.Slider(minimum=5,maximum=100,step=5,label='top_k',value= 40,interactive=True, visible=True)
        num_ctx = gr.Slider(minimum=5,maximum=14,step=1,label='2^num_ctx',value=11,interactive=True, visible=True)
        repeat_penalty = gr.Slider(minimum=0.3,maximum=2,step=0.1,label='repeat_penalty',value=1.1,interactive=True,visible=True)
        seed = gr.Number(label='seed',value=0,visible=True)
        num_gpu = gr.Number(label='num_gpu',value=-1,visible=True)
        stop = gr.Textbox(label="shift+enter to input more stop text. default:AI assistant:", value="AI assistant:", lines=2, visible=True)

    return(num_predict,
            temperature,
            top_p,
            top_k,
            num_ctx,
            repeat_penalty,
            seed,
            num_gpu,
            stop
            )

def save_params(
                num_predict,
                temperature,
                top_p,
                top_k,
                num_ctx,
                repeat_penalty,
                seed,
                num_gpu,
                stop
                ):
    params = {'num_predict':num_predict, 
                'temperature':temperature,
                'top_p':top_p,
                'top_k':top_k,
                'num_ctx':num_ctx,
                'repeat_penalty':repeat_penalty,
                'seed':seed,
                'num_gpu':num_gpu,
                'stop':stop
            }
    with open('.\cache\params.json','w+') as file:
        file.write(json.dumps(params))
    return('params has been saved')
    



    