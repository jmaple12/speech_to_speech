cache_path='./cache/'
tts_api_path = r"E:\LargeModel\Speech_Synthesis\GPT_Sovits\GPT-SoVITS-beta\GPT-SoVITS-beta_fast_inference_0316\api_v2_maple.bat"
import subprocess
import re
import os
import time
class run_cmd():
    def __init__(self, path=cache_path):
        path = path.replace('\\','/')
        if path[-1]=='/':
            path = path[:-1]
        path = os.path.abspath(path)
        path = path.replace('\\','/')
        self.path = path+'/'
        print('模拟CMD运行时cache存放的初始路径为%s'%self.path)

    def find_cmd_pid(self, target_exe='cmd'):
        '''输出在***.exe运行的所有进程的pid'''
        #输出到txt是防止编码错误
        try:
            ss = subprocess.run('tasklist /fi "imagename eq %s.exe"'%target_exe, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE, text=True, encoding='gb18030')
        except:
            ss = subprocess.run('tasklist /fi "imagename eq %s.exe"'%target_exe,stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE, text=True, )
        pid_list = ss.stdout
        temp =re.findall('\d+ Console', pid_list)
        if temp:
            pid_list = re.sub('Console|\n','', ''.join(temp)).strip(' ').split(' ')
            return(pid_list)
        else:
            print("未发现目标程序%s.exe对应的进程"%target_exe)
            return([])
    
    def find_port_pid(self, port):
        '''输出指定端口的所有进行的PId'''
        ss = subprocess.run('netstat -ano | findstr :%d'%int(port),stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE, text=True)
        #查看指定端口的PID信息
        pid_list = ss.stdout
        pid_list = re.sub('\ +', ' ', pid_list).split('\n')[:-1]
        pid_list = [pip.split()[-1] for pip in pid_list]
        return(pid_list)
    
    def close_pid(self, pid_list):
        '''强制关闭指定PID的进程'''
        #如果输入的是数字或者字符串形式的数字，转为单元素的列表
        if not pid_list:
            print("未输入PID，跳过执行关闭进程的程序")
        else:
            if type(pid_list)!=list:
                pid_list = [int(pid_list)]     
            text = ''
            for pid in pid_list:
                if pid !=0:
                    text += "taskkill /PID %d /F\n"%int(pid)
            with open(self.path+'kill.bat', 'w+') as file:
                file.write(text)
            res = subprocess.Popen(['cmd','/c', self.path+'kill.bat'], stdout=open(self.path+'output.txt','w+'), stderr=open(self.path+'error.txt', 'w+'), shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
            result = ' '.join(list(map(lambda x:str(x), pid_list)))
            if result:
                print("进程%s已经关闭"%result)

    def gpt_tts_run(self, tts_bat_path=tts_api_path):
        ''''根据api所在的文件路径，在cache_path下面复制一个修改过的然后运行,不会更改os路径 '''
        #开启前先情况之前的
        if not os.path.exists(tts_bat_path):
            print("输入的文件路径<%s>不存在，请输入正确路径后两次点击重新启动TTS"%tts_bat_path)
        if tts_api_path.split('.')[-1] !='bat':
            print("输入文件不正确，请检查文件路径，输入正确路径后两次点击重新启动TTS")
        self.gpt_tts_close()
        cd_cmd = '\\'.join(tts_bat_path.split('\\')[:-1])
        cd_cmd = '"'+cd_cmd+'"'
        with open(tts_bat_path, 'r', encoding='UTF-8') as file:
            text = file.read()
        text = text.split('\n')
        text = [content for content in text if content!='pause']
        text = ['cd '+cd_cmd]+text
        text = '\n'.join(text)
        with open(self.path+'gpt_tts.bat', 'w+', encoding='UTF-8') as file:
            file.write(text)
        print("开始运行%s.....批处理文件"%tts_bat_path)
        tts = subprocess.Popen(['cmd','/c', self.path+'gpt_tts.bat'], stdout=open(self.path+'tts_output.txt','w+'), stderr=open(self.path+'tts_error.txt', 'w+'), shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
        
    def return_tts_pid(self):
        if not os.path.exists(self.path+'tts_error.txt'):
            print('目标tts_error文件不存在')
            return(None)
        try:
            temp1 = re.search('Started server process \[\d+\]', open(self.path+'tts_error.txt', 'r').read())
        except:
            temp1 = re.search('Started server process \[\d+\]', open(self.path+'tts_error.txt', 'r', encoding="gb18030").read())

        if temp1:
            temp2 = re.search('\d+', temp1.group())
            if temp2:
                return(temp2.group())
        print("未找到进程的PID")
        return(None)
    
    def gpt_tts_close(self):
        pid = self.return_tts_pid()
        if pid:
            self.close_pid(pid)
            #清空tts_error文件
            open(self.path+'tts_error.txt','w+').write('')
            print("成功关闭gpt_tts")
        else:
            print("尝试寻找端口9880的所有进程")
            pid = self.find_port_pid(9880)
            if pid:
                self.close_pid(pid)
                print("成功关闭端口9880的所有进程")
            else:
                print("端口9880无进程")
    
    def cmd_open(exe_path):
        #ollama app路径：r"C:\Users\maple\AppData\Local\Programs\Ollama\ollama app.exe"
        res = subprocess.Popen([exe_path],stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_CONSOLE)
    
    def close_ollama(self):
        self.close_pid(self.find_cmd_pid('ollama'))
        print("ollama相关的进程已关闭")
        self.close_pid(self.find_cmd_pid("ollama_llama_server"))
        print("ollama_llama_server相关的进程已关闭")
        self.close_pid(self.find_cmd_pid('ollama app'))
        print("ollama app相关的进程已关闭")

    def run_ollama(self):
        exe_path = os.path.expanduser('~')+'\\AppData\\Local\\Programs\\Ollama\\ollama app.exe'
        subprocess.Popen([exe_path],stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_CONSOLE)

    def clear_rerun_ollama(self):
        '''重启ollama app'''
        self.close_ollama()
        time.sleep(3)
        import torch
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        del torch
        self.run_ollama()
    
    def close_pid_and_port(self, pid=None, port=None):
        '''关闭指定的端口和线程'''
        if pid:
            self.close_pid(pid)
        if port:
            self.close_pid(self.find_port_pid(port))
        
        


