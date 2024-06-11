import base64
import io
import requests
import numpy as np
from cached_path import cached_path

def wav_to_base64(file_path):
    if not file_path:
        return None
    with open(file_path, "rb") as wav_file:
        wav_content = wav_file.read()
        base64_encoded = base64.b64encode(wav_content)
        return base64_encoded.decode("utf-8")


url = 'http://127.0.0.1:7860'

text = """
Zane glared at you as he passed your open room door. 
He walked to your brothers room to hang out once again. 
Zane flipped you off before closing the door. 
Your brother leaves the room to help with your mother with dinner. 
Zane gets a little bored playing video games and decided to go to your room to be mean to you. 
\"Hey, what are you doing?\"
"""

DEFAULT_TARGET_VOICE_URL = "https://mirror.ghproxy.com/https://raw.githubusercontent.com/styletts2/styletts2.github.io/main/wavs/LJSpeech/OOD/GT/00001.wav"
reference_audio = cached_path(DEFAULT_TARGET_VOICE_URL)
base64_audio = wav_to_base64(reference_audio)

data = {
    "text": text,
    "voice": base64_audio,
    "output_sample_rate": 24000,
    "diffusion_steps": 20,
    "format": "wav",
}

response = requests.post(f"{url}/invoke", json=data)

if response.status_code == 200:
    audio_content = response.content
    with open("generated_audio.wav", "wb") as audio_file:
        audio_file.write(audio_content)
    print("Audio has been saved to 'generated_audio.wav'.")
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.json())


base64_audio = wav_to_base64(reference_audio)
data = {"voice": base64_audio}

response = requests.post(f"{url}/register", json=data)

if response.status_code == 200:
    base64_data = response.json()['embedding']
    binary_data = base64.b64decode(base64_data)
    embedding = np.load(io.BytesIO(binary_data))
    print(embedding.shape, embedding.dtype)
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.json())

data = {
    "text": text,
    "voice": base64_data,
    "output_sample_rate": 24000,
    "diffusion_steps": 20,
    "format": "wav",
}

response = requests.post(f"{url}/generate", json=data)

if response.status_code == 200:
    audio_content = response.content
    with open("generated_audio.wav", "wb") as audio_file:
        audio_file.write(audio_content)
    print("Audio has been saved to 'generated_audio.wav'.")
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.json())