#!/usr/bin/env python3

# Real-time speech recognition from a microphone with sherpa-onnx Python API
# with endpoint detection.
#
# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
# to download pre-trained models

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File
import uvicorn

import argparse
import sys
from pathlib import Path
from stream_zipformer.punct_model import OnnxModel
import time
import os
import numpy as np
try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

import sherpa_onnx


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Path to the encoder model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="Path to the decoder model",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        required=True,
        help="Path to the joiner model",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="Valid values are greedy_search and modified_beam_search",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--hotwords-file",
        type=str,
        default="",
        help="""
        The file containing hotwords, one words/phrases per line, and for each
        phrase the bpe/cjkchar are separated by a space. For example:

        ▁HE LL O ▁WORLD
        你 好 世 界
        """,
    )

    parser.add_argument(
        "--hotwords-score",
        type=float,
        default=1.5,
        help="""
        The hotword score of each token for biasing word/phrase. Used only if
        --hotwords-file is given.
        """,
    )

    parser.add_argument(
        "--blank-penalty",
        type=float,
        default=0.0,
        help="""
        The penalty applied on blank symbol during decoding.
        Note: It is a positive value that would be applied to logits like
        this `logits[:, 0] -= blank_penalty` (suppose logits.shape is
        [batch_size, vocab] and blank id is 0).
        """,
    ),
    parser.add_argument(
        "--flag",
        type=int,
        default=0,
        help="""
        当flag为1时候表示输出check_recognizer,当输出0的时候表示调用recognizer参数进行语音推理。
        """,
    ),
    parser.add_argument("-a", "--host_asr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default="9337", help="default: 9337")
    return parser.parse_args()


def create_recognizer(args):
    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)
    assert_file_exists(args.tokens)
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
    # for download links.
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=3,
        rule2_min_trailing_silence=0.8,
        rule3_min_utterance_length=np.inf,  # it essentially disables this rule
        decoding_method=args.decoding_method,
        provider=args.provider,
        hotwords_file=args.hotwords_file,
        hotwords_score=args.hotwords_score,
        blank_penalty=args.blank_penalty,
    )
    return recognizer

args = get_args()
port = args.port
host = args.host_asr

def main(flag=1, recognizer=None):
    #添加端点的模型
    punct_model = OnnxModel()
    if flag:
        devices = sd.query_devices()
        if len(devices) == 0:
            print("No microphone devices found")
            sys.exit(0)

        #device输出总是出错，如果想要看结果，可以在launch_gradio中使用sounddevice.query_devices()单独输出
        # print(devices)
        default_input_device_idx = sd.default.device[0]
        # print(f'Use default device: {devices[default_input_device_idx]["name"]}')
        recognizer = create_recognizer(args)
        print('machine is found!')
        #asr初始化完成的标志，用于在gradio中作为删除前面无用信息的标志
        print("ASR Launched")
        #1.1秒后删除参数设备信息的记录，减小后面读写记录的压力
        # time.sleep(1.1)
        return(recognizer)

    # print("用户:  请输入你的问题",)
    # os.system('cls')#清屏对tts_err.txt文本记录无用
    begin_word = '用户：'
    # begin_word = ''

    # The model is using 16 kHz, we use 48 kHz here to demonstrate that
    # sherpa-onnx will do resampling inside.
    sample_rate = 48000
    sample_rate = 16000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    stream = recognizer.create_stream()

    last_result = ""
    final_result=""
    segment_id = 0
    #输出到cmd的时候，\r会清空该行内容，把全部内容重新填充，这会导致cmd每一次自动换行的时候，下一次输出时，上一次输出都会重复一行。
    print('\r',end='')
    print(' '*30,end='')
    print('\r',end='')
    print(begin_word,end='')
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            stream.accept_waveform(sample_rate, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            is_endpoint = recognizer.is_endpoint(stream)
            result = recognizer.get_result(stream)
            if result and (last_result != result):
                last_result = result
                print("\r{}".format(begin_word+final_result+result), end="", flush=True)
            if is_endpoint:
                if len(result)>0:
                    #转为带标点的文字--先转为小写，大写英文字符没法添加标点，但是英语自动转写为大写
                    final_result +=punct_model(last_result.lower())
                    final_result +=''
                    print('\r{}'.format(begin_word+final_result),end='', flush=True)
                    segment_id += 1
                else:#是端点且上个端点到这个端点位置无解码，停止录音。
                    # if final_result!=begin_word:
                        # final_result = final_result[:-1]
                        # print('\n', begin_word+final_result, flush=True)
                    print()
                    return(last_result, final_result)
                recognizer.reset(stream)


check_recognizer =main()

APP = FastAPI()
@APP.get("/flag")
def tts_get_endpoint(flag: int=0):
    global check_recognizer
    if flag:
        check_recognizer = main(flag)
        res=''
    else:
        res = main(flag, check_recognizer)
    return res

if __name__ == "__main__":
    try:
        uvicorn.run(APP, host=host, port=port, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
