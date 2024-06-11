from __version__ import __version__

import time
import base64
import os
import io
import magic
import mimetypes
import tempfile
import numpy as np
from datetime import datetime
from typing import Union, Tuple, Annotated
from styletts2 import tts
from pydub import AudioSegment

from pydantic import BaseModel, Field
from fastapi import FastAPI, BackgroundTasks, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import asyncio

import logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_level = log_level.upper()
if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
    log_level = "INFO"
logging.basicConfig(level=getattr(logging, log_level))

host = str(os.getenv("SERVICE_HOST", "0.0.0.0"))
port = int(os.getenv("SERVICE_PORT", "7860"))
workers = int(os.getenv("NUM_WORKERS", "1"))

MAX_MODEL_INSTANCES = int(os.getenv("MAX_MODEL_INSTANCES", "4"))
QUEUE_TIMEOUT_SEC = float(os.getenv("QUEUE_TIMEOUT_SEC", "90.0"))

warmup_text = "This is an inference API for StyleTTS2. It is now warming up..."

def load_model(chkpt_path=None, config_path=None):
    load_start = time.perf_counter()
    model = tts.StyleTTS2(chkpt_path, config_path)
    model.inference(warmup_text)
    logging.info(f"Model loaded in {time.perf_counter() - load_start} seconds.")
    return model

model_instances = [load_model() for _ in range(MAX_MODEL_INSTANCES)]
model_queue = asyncio.Queue(maxsize=MAX_MODEL_INSTANCES)
for i in range(MAX_MODEL_INSTANCES):
    model_queue.put_nowait(i)  # fill the queue with model indices

async def get_model_index():
    try:
        start = time.perf_counter()
        model_index = await asyncio.wait_for(model_queue.get(), timeout=QUEUE_TIMEOUT_SEC)
        return (model_index, time.perf_counter() - start)
    except asyncio.TimeoutError:
        return (None, time.perf_counter() - start)


app = FastAPI()


def get_current_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_client_infos(client):
    return f"{client.host}:{client.port}"

class Timer:
    def __init__(self, path: str, client: str):
        self.path = path
        self.client = client

    def __enter__(self):
        logging.info(f"@@ {get_current_time()} | {self.client} | {self.path} receive request")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.info(f"@@ {get_current_time()} | {self.client} | {self.path} return response")

def set_timer(request: Request):
    with Timer(request.url.path, get_client_infos(request.client)) as timer:
        yield timer


def process_audio(audio_sample: str):
    audio_data = base64.b64decode(audio_sample)
    audio_buffer = io.BytesIO(audio_data)
    mime_type = magic.from_buffer(audio_buffer.read(2048), mime=True)
    file_type = mimetypes.guess_extension(mime_type)[1:]
    audio_buffer.seek(0)
    audio = AudioSegment.from_file(audio_buffer, format=file_type)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio.export(f.name, format="wav")
        return f.name


def get_wav_length_from_bytesio(bytes_io):
    # Ensure the buffer's position is at the start
    bytes_io.seek(0)
    audio = AudioSegment.from_file(bytes_io, format="wav")

    # Calculate the duration in milliseconds, then convert to seconds
    duration_seconds = len(audio) / 1000.0
    return audio, duration_seconds


class RegRequest(BaseModel):
    voice: Union[str, None] = Field(
        default=None,
        title="Base64 encoded voice sample",
        description="Reference voice, the model will attempt to match the voice of the provided sample. 3-10s of sample audio is recommended.",
    )


class GenRequest(BaseModel):
    text: str = Field(..., title="Text to convert to speech")
    voice: Union[str, None] = Field(
        default=None,
        title="Base64 encoded voice sample",
        description="If provided, the model will attempt to match the voice of the provided sample. 3-10s of sample audio is recommended.",
    )
    output_sample_rate: int = Field(
        default=24000, examples=[16000, 22050, 24000, 32000, 44100, 48000],
        title="Output sample rate",
        description="The sample rate of the output audio. Default is 24,000 Hz, and it is suggested to be higher than 16,000 Hz.",)
    alpha: float = Field(
        default=0.3, ge=0.0, le=1.0, strict=True,
        title="Alpha",
        description="`alpha` is the factor to determine much we use the style sampled based on the text instead of the reference. The higher the value of `alpha`, the more suitable the style it is to the text but less similar to the reference. `alpha` determines the timbre of the speaker.",
    )
    beta: float = Field(
        default=0.7, ge=0.0, le=1.0, strict=True,
        title="Beta",
        description="`beta` is the factor to determine much we use the style sampled based on the text instead of the reference. The higher the value of `beta` the more suitable the style it is to the text but less similar to the reference. Using higher beta makes the synthesized speech more emotional, at the cost of lower similarity to the reference. `beta` determines the prosody of the speaker.",
    )
    diffusion_steps: int = Field(
        default=5, ge=3, le=20, strict=True,
        title="Diffusion steps",
        description="Since the sampler is ancestral, the higher the steps, the more diverse the samples are, with the cost of slower synthesis speed.",
    )
    embedding_scale: float = Field(
        default=1, ge=1.0, le=10.0, strict=True,
        title="Embedding scale",
        description="This is the classifier-free guidance scale. The higher the scale, the more conditional the style is to the input text and hence more emotional.",
    )
    output_format: str = Field(
        "wav", examples=["wav", "mp3", "flac"],
        title="Output audio format",
        description="The format of the output audio. Supported formats are 'wav', 'mp3', and 'flac'.",
    )


class TTSRequest(BaseModel):
    text: str = Field(..., title="Text to convert to speech")
    voice: Union[str, None] = Field(
        default=None,
        title="Base64 encoded voice sample",
        description="If provided, the model will attempt to match the voice of the provided sample. 3-10s of sample audio is recommended.",
    )
    output_sample_rate: int = Field(
        default=24000, examples=[16000, 22050, 24000, 32000, 44100, 48000],
        title="Output sample rate",
        description="The sample rate of the output audio. Default is 24,000 Hz, and it is suggested to be higher than 16,000 Hz.",)
    alpha: float = Field(
        default=0.3, ge=0.0, le=1.0, strict=True,
        title="Alpha",
        description="`alpha` is the factor to determine much we use the style sampled based on the text instead of the reference. The higher the value of `alpha`, the more suitable the style it is to the text but less similar to the reference. `alpha` determines the timbre of the speaker.",
    )
    beta: float = Field(
        default=0.7, ge=0.0, le=1.0, strict=True,
        title="Beta",
        description="`beta` is the factor to determine much we use the style sampled based on the text instead of the reference. The higher the value of `beta` the more suitable the style it is to the text but less similar to the reference. Using higher beta makes the synthesized speech more emotional, at the cost of lower similarity to the reference. `beta` determines the prosody of the speaker.",
    )
    diffusion_steps: int = Field(
        default=5, ge=3, le=20, strict=True,
        title="Diffusion steps",
        description="Since the sampler is ancestral, the higher the steps, the more diverse the samples are, with the cost of slower synthesis speed.",
    )
    embedding_scale: float = Field(
        default=1, ge=1.0, le=10.0, strict=True,
        title="Embedding scale",
        description="This is the classifier-free guidance scale. The higher the scale, the more conditional the style is to the input text and hence more emotional.",
    )
    output_format: str = Field(
        "wav", examples=["wav", "mp3", "flac"],
        title="Output audio format",
        description="The format of the output audio. Supported formats are 'wav', 'mp3', and 'flac'.",
    )


@app.get("/health")
def api_health():
    return {"status": "ok", "version": __version__}


@app.post("/register", dependencies=[Depends(set_timer)])
async def api_register(request: Request, api_request: RegRequest,
                       background_tasks: BackgroundTasks,
                       model_index: Tuple[Union[int, None], float] = Depends(get_model_index)):
    model_index, waiting_time = model_index
    if model_index is None:
        return JSONResponse(content={"error": "service is busy."}, status_code=503)
    client_infos = get_client_infos(request.client)
    start = time.perf_counter()
    params = api_request.model_dump()
    wav_buffer = None
    if "voice" not in params or params["voice"] is None:
        return JSONResponse(content={"error": "voice is required."}, status_code=400)
    wav_buffer = process_audio(api_request.voice)
    params["target_voice_path"] = wav_buffer
    del params["voice"]
    try:
        embedding = model_instances[model_index].compute_style(**params).cpu().numpy()
    except Exception as e:
        return JSONResponse(content={"service error": str(e)}, status_code=500)
    model_queue.put_nowait(model_index)  # return the model index to the queue
    inference_time = time.perf_counter() - start
    background_tasks.add_task(os.remove, wav_buffer)
    emb_buffer = io.BytesIO()
    np.save(emb_buffer, embedding)
    binary_data = emb_buffer.getvalue()
    base64_data = base64.b64encode(binary_data).decode("utf-8")
    logging.info(f"@@ {get_current_time()} | {client_infos} | /register " \
                 f"index: {model_index}, time: {waiting_time} {inference_time}")
    return JSONResponse(content={"embedding": base64_data,
                                 "dtype": str(embedding.dtype),
                                 "shape": str(embedding.shape)})


@app.post("/generate", dependencies=[Depends(set_timer)])
def api_generate(request: Request, api_request: GenRequest,
                 model_index: Tuple[Union[int, None], float] = Depends(get_model_index)):
    model_index, waiting_time = model_index
    if model_index is None:
        return JSONResponse(content={"error": "service is busy."}, status_code=503)
    client_infos = get_client_infos(request.client)
    start = time.perf_counter()
    params = api_request.model_dump()
    output_format = params["output_format"]
    del params["output_format"]
    if "voice" in params and params["voice"] is not None:
        binary_data = base64.b64decode(api_request.voice)
        params["ref_s"] = np.load(io.BytesIO(binary_data))
    del params["voice"]
    wav_bytes = io.BytesIO()
    try:
        model_instances[model_index].inference(
            **params,
            output_wav_file=wav_bytes,
        )
    except Exception as e:
        return JSONResponse(content={"service error": str(e)}, status_code=500)
    model_queue.put_nowait(model_index)  # return the model index to the queue
    inference_time = time.perf_counter() - start
    audio, duration_seconds = get_wav_length_from_bytesio(wav_bytes)
    headers = {
        "x-inference-time": str(inference_time),
        "x-audio-length": str(duration_seconds),
        "x-realtime-factor": str(inference_time / duration_seconds),
    }
    return_bytes = io.BytesIO()
    audio.export(return_bytes, format=output_format)
    return_bytes.seek(0)
    logging.info(f"@@ {get_current_time()} | {client_infos} | /generate " \
                 f"index: {model_index}, time: {waiting_time} {inference_time}")
    return StreamingResponse(
        return_bytes,
        media_type=f"audio/{output_format}",
        headers=headers,
    )


@app.post("/invoke", dependencies=[Depends(set_timer)])
def api_invoke(request: Request, api_request: TTSRequest,
               background_tasks: BackgroundTasks,
               model_index: Tuple[Union[int, None], float] = Depends(get_model_index)):
    model_index, waiting_time = model_index
    if model_index is None:
        return JSONResponse(content={"error": "service is busy."}, status_code=503)
    start = time.perf_counter()
    params = api_request.model_dump()
    output_format = params["output_format"]
    del params["output_format"]
    wav_buffer = None
    if "voice" in params and params["voice"] is not None:
        wav_buffer = process_audio(api_request.voice)
        params["target_voice_path"] = wav_buffer
    del params["voice"]
    register_time = time.perf_counter() - start
    wav_bytes = io.BytesIO()
    try:
        model_instances[model_index].inference(
            **params,
            output_wav_file=wav_bytes,
        )
    except Exception as e:
        return JSONResponse(content={"service error": str(e)}, status_code=500)
    generate_time = time.perf_counter() - start - register_time
    inference_time = register_time + generate_time
    model_queue.put_nowait(model_index)  # return the model index to the queue
    audio, duration_seconds = get_wav_length_from_bytesio(wav_bytes)
    if wav_buffer is not None:
        background_tasks.add_task(os.remove, wav_buffer)
    headers = {
        "x-inference-time": str(inference_time),
        "x-audio-length": str(duration_seconds),
        "x-realtime-factor": str(inference_time / duration_seconds),
    }
    return_bytes = io.BytesIO()
    audio.export(return_bytes, format=output_format)
    return_bytes.seek(0)
    logging.info(f"@@ {get_current_time()} | {get_client_infos(request.client)} | /invoke " \
                 f"index: {model_index}, time: {waiting_time} {register_time} {generate_time}")
    return StreamingResponse(
        return_bytes,
        media_type=f"audio/{output_format}",
        headers=headers,
    )


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port, workers=workers)
