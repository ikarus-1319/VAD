import sys

sys.path.append('../')

import pyaudio
import wave
from s2t import s2t_zalo
import sounddevice as sd
import wavio
import io
from google.cloud import speech_v1p1beta1 as speech
from pydub import AudioSegment
import re
import os

def preprocess_audio(audio_file_path, target_sample_rate=16000):
    # Load the audio file
    audio = AudioSegment.from_file(audio_file_path)
    
    # Convert to mono
    audio = audio.set_channels(1)
    
    # Set the sample rate to 16000 Hz
    audio = audio.set_frame_rate(target_sample_rate)
    
    # Export as WAV with LINEAR16 encoding
    processed_audio_path = "processed_audio.wav"
    audio.export(processed_audio_path, format="wav", codec="pcm_s16le")
    
    return processed_audio_path
 
def transcribe_audio(audio_file_path):
    client = speech.SpeechClient()
 
    with io.open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()
 
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="vi-VN",
    )
 
    response = client.recognize(config=config, audio=audio)
 
    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript + " "
    
    transcript = transcript.strip()
    print("Transcript: {}".format(transcript))
    return transcript

config_zalo =  {
    "asr_modelname": "zalo",
    "asr_api_url": "https://api.zalo.ai/telio_asr_offline",
    "asr_api_headers": {
        "apikey": "7ka2xzSWiOnomrtmdykvweFnPXb4sMq4"
    }
}

# Các tham số ghi âm
SAMPLE_RATE = 16000  # Tần số lấy mẫu (samples per second)
DURATION = 10  # Thời gian ghi âm (giây)
OUTPUT_FILENAME1 = "/home/pc-trunghieu-20/code/silero-vad/audio_1.wav"
OUTPUT_FILENAME2 = "/home/pc-trunghieu-20/code/silero-vad/audio_filtered_1.wav"
OUTPUT_FILENAME3 = "/home/pc-trunghieu-20/code/silero-vad/handle_audio.wav"
OUTPUT_FILENAME4 = "/home/pc-trunghieu-20/code/silero-vad/audio.wav"
# print("Bắt đầu ghi âm...")

# # Ghi âm
# recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
# sd.wait()  # Đợi cho đến khi ghi âm xong

# print("Ghi âm xong!")

# # Lưu dữ liệu vào file WAV
# wavio.write(OUTPUT_FILENAME, recording, SAMPLE_RATE, sampwidth=2)

# print(f"Dữ liệu đã được lưu vào file {OUTPUT_FILENAME}")

# transcript = s2t_zalo(config_zalo, OUTPUT_FILENAME1)
# print(transcript)
# transcript = s2t_zalo(config_zalo, OUTPUT_FILENAME2)
# print(transcript)
transcript = s2t_zalo(config_zalo, OUTPUT_FILENAME3)
print(transcript)
transcript = s2t_zalo(config_zalo, OUTPUT_FILENAME4)
print(transcript)