FROM vemlp-cn-shanghai.cr.volces.com/preset-images/pytorch:2.1.0-cu11.8.0-py3.10-ubuntu20.04

WORKDIR /app
ENV TZ=etc/UTC DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
  git \
  build-essential \
  ffmpeg \
  libmagic1

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
COPY resource /app/resource
RUN pip install -r requirements.txt -i https://mirrors.ivolces.com/pypi/simple

ENV NLTK_DATA=/app/resource/nltk_data
RUN python -c "from styletts2 import tts; tts.StyleTTS2()"

COPY app/ .

EXPOSE 7860

# CMD ["python", "server.py"]