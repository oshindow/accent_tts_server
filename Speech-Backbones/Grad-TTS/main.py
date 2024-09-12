import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy import io, signal
import numpy as np
from PIL import Image
import librosa
from io import BytesIO
import PIL.Image
# import cv2
import os
from librosa.filters import mel as librosa_mel_fn
import torch
import torchaudio
import random
import shutil
import requests
import soundfile as sf
import pyworld as pw
ORI_DICTIONARY = "/data1/jiyuyu/AISHELL-1/data_aishell/wav/train/S0002/"
WAV_DICTIONARY = "./wav/"
PNG_DICTIONARY = "./png"
SAMPLE_RATE = 16000

mel_basis = {}
hann_window = {}
n_fft=1024
num_mels=80
sampling_rate=16000
hop_size=160
win_size=1024
fmin=0
fmax=300

def wav2array(wav, fs):
    sample_rate, data = io.wavfile.read(wav)
    # 而 对 楼市 成交 抑制 作用 最 大 的 限 购

    f, t, z = signal.stft(data, fs=fs)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, np.abs(z)**2, shading='gouraud')
    plt.savefig('ori.png')
    # plt.plot(t, f, color='purple', linestyle='-')
    plt.ylim(20, 500)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    img = PIL.Image.open(buf)
    img_array = np.array(img)

    return img_array

def dynamic_range_compression_torch_log10(x, C=1, clip_val=1e-5):
    return torch.log10(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch_log10(magnitudes):
    output = dynamic_range_compression_torch_log10(magnitudes)
    return output

def mel_spectrogram_align(y, n_fft=n_fft, num_mels=num_mels, sampling_rate=sampling_rate, hop_size=hop_size, win_size=win_size, fmin=fmin, fmax=fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    # y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    # y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

    # spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    spec = torch.sqrt(spec.real.pow(2).add(spec.imag.pow(2)) + 1e-9)

    ### mel-scale
    # spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch_log10(spec)

    # freq = librosa.mel_frequencies(n_mels=num_mels, fmin=fmin, fmax=fmax)
    freq = librosa.fft_frequencies(n_fft=n_fft, sr=sr)
    # print(torch.max(spec), torch.min(spec))
    return spec, freq

def plot(waveform, sr, spectrogram, freq, savename, timestamp=None, pitches=None):
    waveform = waveform.numpy()
    spectrogram = spectrogram.squeeze().numpy()

    hop_size = 256
    # for start, end in timestamp:
    #     start_time = start / 1000 / hop_size * sr
    #     end_time = end / 1000 / hop_size * sr
    #     plt.axvline(x=start_time, color='red', linestyle='-', linewidth=1)
    #     plt.axvline(x=end_time, color='blue', linestyle='-', linewidth=1)
    # freq = freq[:64]
    plt.imshow(spectrogram, origin="lower", aspect="auto", interpolation="nearest", extent=[0, spectrogram.shape[1], freq[0], freq[-1]])
    plt.xlabel("time")
    plt.ylabel("Hz")
    
    plt.xlim(0, spectrogram.shape[1])
    plt.ylim(freq[0], freq[-1])
    print(pitches)
    # pitches = pitches[200:700]
    # for idx in range(len(pitches)):
    #     pt = pitches[idx]
    #     plt.plot([idx], [pt], color='r', marker='o', markersize=0.5, linewidth=2)  # Use marker='o' to show points clearly

        # plt.scatter(idx, pt, color='black', s=5)

    plt.savefig(f'{savename}.png', dpi=1000)
    plt.close()

def select_random_files(dir, num):
    file_list = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    if len(file_list) < num:
        raise ValueError()
    selected_files = random.sample(file_list, num)
    return [os.path.join(dir, f) for f in selected_files]

def copy_files(copy_list, wav_dir):
    for file in copy_list:
        shutil.copy(file, wav_dir)
    new_file_list = [f for f in os.listdir(wav_dir) if os.path.isfile(os.path.join(wav_dir, f))]
    return new_file_list

def transcribe(my_files):
    response = []
    try:
        url = "http://localhost:8000/recognize"
        files = {"file": open(my_files, "rb")}
        response = requests.post(url, files=files)
    except:
        print('Cannot connect to api server.')
    return response.json()

def get_duration(audio):
    waveform, _ = torchaudio.load(audio)
    duration = waveform.size(1) / sampling_rate
    return duration

def select_audio_withmins(ori_dir, max_duration):
    file_list =[f for f in os.listdir(ori_dir) if os.path.isfile(os.path.join(ori_dir, f))]
    selected_list = []
    total_duration = 0
    while total_duration < max_duration and file_list:
        selected_file = random.choice(file_list)
        file_list.remove(selected_file)
        selected_file = os.path.join(ori_dir, selected_file)
        selected_list.append(selected_file)
        print("select ", selected_file)
        total_duration += get_duration(selected_file)
    return selected_list

def get_pitch(response):
    pitches = []
    length = len(response["characters"])
    characterTimestamps = response["characterTimestamps"]
    pitch_list = response["pitches"]

    for index, timestamp in enumerate(characterTimestamps):
        start, end = timestamp
        start = start / 1000 / hop_size * sr
        end = end / 1000 / hop_size * sr
        step = (end - start) // len(pitch_list[index])
        for s, pitch in enumerate(pitch_list[index]):
            pitches.append([start + s * step, pitch])
    # print(pitches)
    return pitches

if __name__ == "__main__":
    # file_list = copy_files(ORI_DICTIONARY, WAV_DICTIONARY, 1)
    # audio_list = select_audio_withmins(ORI_DICTIONARY, 3600)
    # file_list = copy_files(audio_list, WAV_DICTIONARY)
    file_list = ["./911_128684_000025_000005.wav"]
    for filename in file_list:
        if filename.endswith('.wav'):
            file_path = os.path.join(WAV_DICTIONARY, filename)
            audio, sr = torchaudio.load(file_path)
            # response = transcribe(file_path)
            spectrogram, freq = mel_spectrogram_align(audio)
            save_path = os.path.join(PNG_DICTIONARY, filename[:-4])
            # pitches = get_pitch(response)
            # x, outFs = sf.read(file_path)
            # pitches = []
            # with open('ref/ref_F01_sa1.f0', 'r', encoding='utf8') as input:
            #     for line in input:
            #         pitch = line.strip().split(' ')[0]
            #         pitches.append(float(pitch))
            # pitches = np.array(pitches)
            f0, t = pw.dio(audio.squeeze(0).numpy().astype(np.float64), sr, frame_period=10)
            pitches = pw.stonemask(audio.squeeze(0).numpy().astype(np.float64), f0, t, sr)

            plot(audio, sr, spectrogram, freq, save_path, pitches=pitches)
