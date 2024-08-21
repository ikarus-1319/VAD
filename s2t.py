import json
import requests
import os
 
def s2t_zalo(config, voice_filepath):
    audio_binary = open(os.path.abspath(voice_filepath), 'rb')
    payload = {'type': 'wav'}
    files = [
        ('byte_data', (voice_filepath, audio_binary, 'audio/wav'))
    ]
 
    response = requests.request(
        method='POST', url=config.get('asr_api_url'), headers=config.get('asr_api_headers'), data=payload, files=files)
    transcript = json.loads(response.text)['data']
 
    return transcript