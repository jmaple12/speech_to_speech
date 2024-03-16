
import pyaudio,wave
import sys
import numpy as np
def listen(WAVE_OUTPUT_FILENAME, tag, delayTime=2, tendure=2, mindb = 500):
    '''
    delaytime:说话声音小持续多少秒就结束；----超过delaytime时候本波次录音结束
    tendure:完全空白录音忍耐时长，超过这个时间结本轮录音结束
    mindb:接受的最低音量
    '''
    sign = 0 #判断是否持续空白输入，若是，sign=1，若声音从大道持续小，sign=2
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 2
    # WAVE_OUTPUT_FILENAME = outfile_name
    # mindb = 500  # 最小声音，大于则开始录音，否则结束
    # delayTime = 1.5  # 小声1.5秒后自动终止
    tendure = tendure*RATE/CHUNK/1.1#大致时长计算转换
    delayTime = delayTime*RATE/CHUNK/1.1
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    if tag==0:
        print("用户:  请输入你的问题",end='')
    frames = []
 
    flag = False  # 开始录音节点
    stat = True  # 判断是否继续录音
    stat2 = False  # 判断声音小了
    tempnum = 0  # tempnum、tempnum2、tempnum3为时间
    tempnum2 = 0

    while stat:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.short)
        temp = np.max(audio_data)
        # print(temp)
        if temp > mindb and not flag:
            flag = True
            # print("开始录音")
            tempnum2 = tempnum

        if flag:
            if temp < mindb and not stat2:
                stat2 = True
                tempnum2 = tempnum
                # print("声音小，且之前是大的或刚开始，记录当前点")

            if temp > mindb:
                stat2 = False
                tempnum2 = tempnum  # 刷新

            if tempnum > tempnum2 + delayTime and stat2:
                # print("间隔%.2lf秒后开始检测是否还是小声" % delayTime)
                if stat2 and temp < mindb:
                    stat = False  # 还是小声，则stat=True
                    sign=2#表示用户有短时间的说话间隔
                    if tag==0:
                        print('\r', end='')
                        print(' '*20,end='')
                        print('\r',end='')
                        # sys.stdout.flush()
                        print('用户: ',end='')
                    tag=1 #标志用户在一轮对话的开头
                    # print("小声！")
                    # print("录音结束")
                else:
                    stat2 = False
                    # print("大声！")

        tempnum += 1
        if tempnum-tempnum2 > tendure and not flag:  #持续的空白就退出且不转语音
        # if tempnum>150:
            stat = False
            sign=1
            # print("录音结束")
    ##检查输出
    # print(sign, tempnum-tempnum2, 'tendure',tendure,'delayTime', delayTime)
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # stream.stop_stream()
    # stream.close()
    # p.terminate()
    # wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    # wf.setnchannels(CHANNELS)
    # wf.setsampwidth(p.get_sample_size(FORMAT))
    # wf.setframerate(RATE)
    # wf.writeframes(b''.join(frames))
    # wf.close()
    return(sign, tag)
