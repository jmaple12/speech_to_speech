chcp 65001
::上面一句表明使用utf-8编码，解决批处理无法识别中文路径的问题
runtime\python.exe api_v2_maple.py
::-g "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt" -s "GPT_SoVITS/pretrained_models/s2G488k.pth" 
::-s "E:\LargeModel\Speech_Synthesis\GPT_SOVITS_0训练集\自己的模型\姬如千泷\训练集_denoise\yueer_denoise_e25_s575.pth" -g "E:\LargeModel\Speech_Synthesis\GPT_SOVITS_0训练集\自己的模型\姬如千泷\训练集_denoise\yueer_denoise-e25.ckpt" ::中文路径不识别
pause