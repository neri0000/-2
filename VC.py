from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from scipy.io.wavfile import write
import copy
import wave

import librosa
import soundfile as sf
 
def Auto_amp_coefficient(original_data, edited_data):
    sampling_num = 48000 #サンプル数．増やすと精度が上がる
    sample = np.random.randint(0, len(original_data), sampling_num)
    amp = max(original_data)/max(edited_data)
    return amp
 

def filter(wave,lf,hf,input,output):

    # ファイルの読み込み
    sourceAudio = AudioSegment.from_wav(input)
    w = wave.open(input, 'rb')
    fs = w.getframerate()
    AudioLength = sourceAudio.duration_seconds
    FrameRate = sourceAudio.frame_rate
    wave = sourceAudio.get_array_of_samples()
    N = len(wave)
    dt = 1/FrameRate/2 
 
    fft = fftpack.fft(wave)                # FFT
    # 元データの隔離,実際に使うのはfft
    fft_original = copy.copy(fft)

    ac = 4  #振幅下限
    #サンプリング周期
    samplerate = N / AudioLength
    fft_axis = np.linspace(0, samplerate, N)    # 周波数軸
    fft_amp = np.abs(fft / (N / 2))  # 振幅


    #上下のカット
    #fft[(fft_amp < ac)] = 0
    fft[(fft_axis < lf)] = 0
    fft[(fft_axis > hf)] = 0
    #fft[(fft_amp < ac)] = 0

    # IFFT処理
    ifft_time = fftpack.ifft(fft) 


    amp = Auto_amp_coefficient(wave, ifft_time.real)
    ifft_time.real *= amp
    audio_data = ifft_time
    data = [[0 for x in range(2)] for i in range(len(audio_data))]
    for i in range(len(ifft_time)):
        data[i][0] = audio_data[i]
        data[i][1] = audio_data[i]
    
    data = np.array(data, dtype='int16')
    
    write(output, 2*FrameRate, data)
 
#VC-------------------------------------------------------------------------


def stretch(data,rate=1):
    input_length = 240000
    data=librosa.effects.time_stretch(data,rate)
    return data

def pitch(data,sr,pitch_factor):
    return librosa.effects.pitch_shift(data,sr,pitch_factor)

def load_audio_file(file_path):
    input_length = 240000
    data = librosa.core.load(file_path,sr=48000)[0] #, sr=16000
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data


sr=48000

#一次フィルタ-------------------------------------------------------------------------------------------
filter(wave,100,5000,"criminal.wav","filtered.wav")  #(wave,lf,hf,wavファイル名)    lf:㎐下限  hf:㎐上限

#VC-----------------------------------------------------------------------------------------------------
data = load_audio_file("filtered.wav")
pitch_data=pitch(data,sr,5.0)
# wavファイル保存
sf.write("filtered-VC.wav", pitch_data, sr, subtype="PCM_16")

#二次フィルタ--------------------------------------------------------------------------------------------
#filter(wave,100,5000,"filtered-VC.wav",filtered-VC-F.wav)
