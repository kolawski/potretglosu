ARG BASE=ghcr.io/coqui-ai/tts
FROM ${BASE} AS builder

#--------------------------------------
# Install dependencies
RUN apt-get update && \
    apt-get install -y git git-lfs \
    alsa-utils \
    ffmpeg \
    pulseaudio && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

#--------------------------------------
# Install python packages
RUN pip install \
    crepe==0.0.16 \
    dask==2022.12.1 \
    dash==2.18.1 \
    onnx==1.12.0 \
    playsound==1.2.2 \
    plotly==5.24.1 \
    praat-parselmouth==0.4.5 \
    pydub==0.25.1 \
    pyarrow==10.0.1 \
    seaborn==0.13.2 \
    torchmetrics==1.6.0
    
#--------------------------------------
# Clean up
RUN rm -rf /root/.cache/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#--------------------------------------
# Download resources
ARG RESOURCES_DIR=/image_resources
WORKDIR ${RESOURCES_DIR}

RUN git clone https://huggingface.co/coqui/XTTS-v2 ${RESOURCES_DIR}/models/XTTS-v2

#--------------------------------------
# Build final image
FROM ${BASE}

ARG RESOURCES_DIR=/image_resources
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder ${RESOURCES_DIR} ${RESOURCES_DIR}

# Dodanie PYTHONPATH dla aplikacji
ENV PYTHONPATH="/app:$PYTHONPATH"
