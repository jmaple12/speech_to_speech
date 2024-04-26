cache_path = './cache/'
import subprocess,os,re
import requests
import time
cache_path = os.path.abspath(cache_path)+'/'
path = "E:\LargeModel\kaldi\sherpa_onnx_model\sherpa_onnx_speech_recognizier.bat"

def asr_init(path=path):
    path = path.replace('\\','/')
    text = open(path, 'r').read()
    text = 'chcp 65001\n'+"cd /d "+'/'.join(path.split('/')[:-1])+'\n'+text
    text = text.replace('\npause','')
    open(cache_path+'sherpa_onnx_speech_recognizier_copy.bat','w+').write(text)
    #不能使用subprocess.run，它只有运行完成前都是阻塞通道，Popen不会。
    asr_sherpa_onnx = subprocess.Popen(['cmd','/c',cache_path+'sherpa_onnx_speech_recognizier_copy.bat'], stderr=open(cache_path+'asr_out.txt','w+',), stdout=open(cache_path+'asr_err.txt','w+',), text=True, creationflags=subprocess.CREATE_NEW_CONSOLE)

def asr_request():
    url = 'http://127.0.0.1:9337/flag'
    res = requests.get(url,)
    res = eval(res.content.decode(encoding='utf-8'))[1]
    return(res)

def check_asr():
    '''检查asr是否正常启动，找出asr_out.txt中的pid和url, 若两者都为None，则出错
    '''
    for i in range(60):
        time.sleep(1)
        temp = open(cache_path+'asr_out.txt','r',).read()
        if temp:
            asr_pid = re.search("Started server process \[\d+\]", temp)
            if asr_pid:
                asr_pid = re.search('\d+', asr_pid.group()).group()
            else:
                asr_pid = None
            asr_url =  re.search("https?://\d+\.\d+\.\d+\.\d:\d+", temp)
            if asr_url:
                asr_url = asr_url.group()
            else:
                asr_url = None

            open(cache_path+'asr_err.txt','w+').write('')
            return(asr_pid, asr_url)
    print("线程和地址未显示")
    return(None, None)

def get_asr_process():
    '''待办：让asr_text_process变更时也运行此程序'''
    with open(cache_path+'asr_err.txt','r',) as file:
        temp = file.read()
        temp = temp.lstrip('\n').lstrip(' ')
        temp2 = re.search('用户：.*\n', temp)
        if temp2:
            temp_phrase = temp2.group()[3:-1]
            open(cache_path+'asr_err.txt','w+',).write(temp[temp2.span()[1]:])
            return(temp_phrase)
        else:
            return(None)
        

