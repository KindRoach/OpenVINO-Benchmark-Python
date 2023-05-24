FROM ubuntu:latest
LABEL authors="kindroach"

RUN apt-get update \
    && apt-get install -y build-essential wget git ffmpeg numactl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PY_ENV="ov-benchmark"

RUN ${CONDA_DIR}/bin/conda create --name ${PY_ENV} python=3.9 -y
ENV PATH=/opt/conda/envs/${PY_ENV}/bin:$PATH

RUN git clone https://github.com/KindRoach/OpenVINO-Python-Benchmark.git
WORKDIR /OpenVINO-Python-Benchmark

RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get install -y ffmpeg

RUN python prepare_data_and_model.py


