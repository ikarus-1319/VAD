from pydub import AudioSegment
import wavio
import sounddevice as sd
import numpy as np
import torch
from silero_vad import get_speech_timestamps
import time
import requests
from Noise_Reduction import NoiseReducer
import scipy.signal
import librosa

def process_audio(data):
    audio_np = np.frombuffer(data, dtype=np.int16)
    return audio_np.astype(np.float32) / 32768.0 

repo_dir = '/home/pc-trunghieu-20/.cache/torch/hub/snakers4_silero-vad_master'
model, utils = torch.hub.load(repo_or_dir=repo_dir, model='silero_vad', source='local')
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

rate = 16000
chunk_duration = 0.5
output_file = '/home/pc-trunghieu-20/code/silero-vad/audio.wav'

frames = []
no_speech_count = 0
max_no_speech_chunks = int(1.5/ chunk_duration)

print("Bắt đầu ghi âm ...")
try:
    start_time = time.time()
    
    while True:
        record = sd.rec(int(chunk_duration * rate), samplerate=rate, channels=1, dtype='int16')
        sd.wait()
        frames.append(record)
        
        audio_data = process_audio(record)
        
        start_handle_record_time = time.time()
        speech_timestamps = get_speech_timestamps(audio_data, model)
        handle_record_time = time.time() - start_handle_record_time
        print(handle_record_time)
        
        if not speech_timestamps:
            no_speech_count += 1
            print('không có âm thanh')
        else:
            no_speech_count = 0  
            print('có âm thanh')
        
        if no_speech_count > max_no_speech_chunks and (time.time() - start_time) >= 3.0:
            print("Không phát hiện giọng nói trong 1.5 giây, dừng ghi âm.")
            break
        
    end_time = time.time()
    
    recorded_audio = np.concatenate(frames)
    wavio.write(output_file, recorded_audio, rate, sampwidth=2)
    print("Đã lưu file ghi âm.")

finally:
    sd.stop()
