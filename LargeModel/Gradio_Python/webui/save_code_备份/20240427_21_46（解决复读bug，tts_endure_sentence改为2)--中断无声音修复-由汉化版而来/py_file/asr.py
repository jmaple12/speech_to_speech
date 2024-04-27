cache_path = './cache/'
import subprocess,os,re
import requests
import time
cache_path = os.path.abspath(cache_path).replace('\\','/')+'/'
path = "E:\LargeModel\kaldi\sherpa_onnx_model\sherpa_onnx_speech_recognizier.bat"

def asr_init(path=path):
    path = path.replace('\\','/')
    text = open(path, 'r').read()
    text = 'chcp 65001\n'+"cd /d "+'/'.join(path.split('/')[:-1])+'\n'+text
    text = text.replace('\npause','')
    open(cache_path+'sherpa_onnx_speech_recognizier_copy.bat','w+').write(text)
    #不能使用subprocess.run，它只有运行完成前都是阻塞通道，Popen不会。
    asr_sherpa_onnx = subprocess.Popen(['cmd','/c',cache_path+'sherpa_onnx_speech_recognizier_copy.bat'], stderr=open(cache_path+'asr_out.txt','w+',), stdout=open(cache_path+'asr_err.txt','w+',), text=True, creationflags=subprocess.CREATE_NEW_CONSOLE, shell=True)

def asr_request():
    url = 'http://127.0.0.1:9337/flag'
    res = requests.post(url,json={})
    res = eval(res.content.decode(encoding='utf-8'))[1]
    print("语音转写的结果为：%s"%res)
    return(res)

def check_asr():
    '''检查asr是否正常启动，找出asr_out.txt中的pid和url, 若两者都为None，则出错
    '''
    for i in range(60):
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
                # print("asr_api端口重复，请关闭api并重新开启")

            open(cache_path+'asr_err.txt','w+').write('')
            return(asr_pid, asr_url)
        time.sleep(1)
    print("线程和地址未显示")
    return(None, None)

def check_asr_duply():
    temp =None
    if os.path.exists(cache_path+'asr_out.txt'):
        temp = open(cache_path+'asr_out.txt', 'r').read()
        if temp:
            temp = re.search("\(\'\d+\.\d+\.\d+\.\d\',(\ )*\d+", temp)
            if temp:
                temp = temp.group().split(',')[-1].strip(' ')
    return(temp)
            


def get_asr_process(input_num):
    '''待办：让asr_text_process变更时也运行此程序
    从input_num开始搜索文本，保存第一个用户：..\n的字段，input_num改为搜索的句尾位置。
    '''
    num =input_num
    with open(cache_path+'asr_err.txt','r',encoding='gb18030') as file:
        temp = file.read()

        #出现info时删除之前的文字，减少search量
        ss = re.search("INFO:.*\n", temp)
        if ss:
            open(cache_path+'asr_err.txt','w+',encoding='gb18030').write(temp[ss.span()[1]:])
            return(None, 0)
        
        temp = temp.lstrip('\n ')
        temp2 = re.search('用户：.*\n', temp[num:])
        if temp2:
            temp_phrase = temp2.group()[3:].strip('\n ')
            num = temp2.span()[1]+num
            print("get_asr_process:", temp_phrase,'||')
            return(temp_phrase, num)
        else:
            return(None, num)


def read_ast_out():
    if os.path.exists(cache_path+'asr_out.txt'):
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
        pid = asr_pid
        url = asr_url
        if url:
            if ":" in url:
                port = url.split(':')[-1]
                print("read asr out的pid,port分别为", pid, port)
                return(pid, port)
        else:
            port = check_asr_duply()
            if port:
                print("read asr out的pid,port分别为", pid, port)
                return(pid, port)
        print("read asr out的pid为", pid)
        return(pid, None)
    return(None, None)
    
        

