# hash:sha256:6eaef11f4d48778aa03d4c9c2a6a24c13f98c41c80a3bf977806dd5b416cae69
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONWARNINGS=ignore::DeprecationWarning

RUN pip3 install -U --no-cache-dir \
    matplotlib==3.9.4 \
    opendssdirect-py==0.9.4 \
    ray[rllib]==2.37.0 \
    torch==2.6.0
