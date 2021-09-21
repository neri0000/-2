import numpy as np
import random
import itertools
import librosa
import IPython.display as ipd
import scipy.io.wavfile as wav
import soundfile as sf

def stretch(data,rate=1):
    input_length = 480000
    data=librosa.effects.time_stretch(data,rate)
    return data
def pitch(data,sr,pitch_factor):
    return librosa.effects.pitch_shift(data,sr,pitch_factor)

def load_audio_file(file_path):
    input_length = 480000
    data = librosa.core.load(file_path,sr=18000)[0] #, sr=16000
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data

sr=18000
data = load_audio_file("serihu.wav")
pitch_data=pitch(data,sr,-3.0)

sf.write('chenge2.wav', pitch_data,sr)

#wavファイル読み込み
fs1, x = wav.read("chenge2.wav")
fs2, h = wav.read("p.wav")
x = x / 32768
h = h / 32768

#フィルタ設計
filter = sig.firls(13, (0, 8000, 12000, 24000), 
                   (1, 1, 0, 0), fs = 48000)

#畳み込み
y = sig.convolve(x, h)
y_int = (y * 32768).astype('i2')

#wavファイル保存
wav.write("chenge4.wav", fs, y_int)