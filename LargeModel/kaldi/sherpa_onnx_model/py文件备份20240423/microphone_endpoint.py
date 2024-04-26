#!/usr/bin/env python3

# Real-time speech recognition from a microphone with sherpa-onnx Python API
# with endpoint detection.
#
# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
# to download pre-trained models

import argparse
import sys
from pathlib import Path
import numpy as np
from stream_zipformer.punct_model import OnnxModel
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


def get_args(args_dict=None):#删除

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
    )

    return parser.parse_args()


def create_recognizer(args_dict, sherpa_params):
    assert_file_exists(args_dict['encoder'])
    assert_file_exists(args_dict['decoder'])
    assert_file_exists(args_dict['joiner'])
    assert_file_exists(args_dict['tokens'])
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
    # for download links.
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=args_dict['tokens'],
        encoder=args_dict['encoder'],
        decoder=args_dict['decoder'],
        joiner=args_dict['joiner'],
        num_threads=6,
        # sample_rate=32000,
        sample_rate = 16000,#176行左右的sample_rate也要对应更改，感觉16000效果挺好
        feature_dim=80,
        enable_endpoint_detection=True,
        # rule1_min_trailing_silence=3,#2.4,#完全静默超过这个时长视为端点 
        # rule2_min_trailing_silence=0.6,#1.2,已经接解码声音了，若静默超过这个时长，视为端点
        rule3_min_utterance_length=np.inf,#300,  # it essentially disables this rule
        decoding_method='greedy_search',#args.decoding_method,
        # provider='cuda',#args.provider,
        hotwords_file='', #args.hotwords_file,
        hotwords_score=1.5, #args.hotwords_score,
        blank_penalty=0, #args.blank_penalty,
        **sherpa_params,
    )
    return recognizer

def main(args, flag=True, recognizer=None, sherpa_params={}):
    #添加端点的模型
    punct_model = OnnxModel()
    # args = get_args(args_dict)
    '''
    flag表示是否检查设备并创建识别模型，如果flag=1,则字直接使用传入的recognizer，此时recognizer！=None
    返回最后一个endpoint的语音结果
    '''
    if flag:
        devices = sd.query_devices()
        if len(devices) == 0:
            print("No microphone devices found")
            sys.exit(0)

        print(devices)
        default_input_device_idx = sd.default.device[0]
        print(f'Use default device: {devices[default_input_device_idx]["name"]}')

        recognizer = create_recognizer(args, sherpa_params=sherpa_params)
        print('machine is found!')
        return(recognizer)

    print("用户:  请输入你的问题", end='')
    begin_word = '用户：'

    # The model is using 16 kHz, we use 48 kHz here to demonstrate that
    # sherpa-onnx will do resampling inside.
    sample_rate = 48000
    sample_rate = 16000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    stream = recognizer.create_stream()

    last_result = ""
    final_result=""
    segment_id = 0

    print('\r',end='')
    print(' '*30,end='')
    print('\r',end='')
    print('用户：',end='')
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
                    #转为带标点的文字
                    final_result +=punct_model(last_result)
                    final_result +=' '
                    print('\r{}'.format(begin_word+final_result),end='', flush=True)
                    segment_id += 1
                else:#是端点且上个端点到这个端点位置无解码，停止录音。
                    if final_result!=begin_word:
                        final_result = final_result[:-1]
                        print('\r{}{}'.format(begin_word+final_result,' '*30), end='', flush=True)
                    return(last_result, final_result)
                recognizer.reset(stream)
