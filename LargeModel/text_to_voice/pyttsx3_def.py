import pyttsx3
#参数参考了https://blog.csdn.net/cui_yonghua/article/details/134611001
engine = pyttsx3.init() # object creation

#记录当前语音的速率，音量和音色
(r,ve,vs) = (engine.getProperty('rate'),engine.getProperty('volume'), engine.getProperty('voices'))
# print(r,ve)
def text_voice(engine, rate=-1, volume=-1, voice=-1):
    '''
    三个参数：
    rate:速率
    volume:音量，【0,1】
    voice:声音，0为男声，1为女声
    '''
    if rate!=-1:
        engine.setProperty('rate', rate)
    if volume!=-1:
        engine.setProperty('volume',volume)
    if voice!=-1:
        voices = engine.getProperty('voices')       
        engine.setProperty('voice', voices[voice].id)
    return engine
    
# engine = text_voice(engine,rate=130)

# engine.say("1. 打开窗户或门，让冷空气进入房间。")
# engine.runAndWait()
# engine.stop()

# """Saving Voice to a file"""
# engine.save_to_file("Hello World!我叫金伍珑,今天也是美好的一天呢！", 'test.mp3')
# engine.runAndWait()
# engine.stop()