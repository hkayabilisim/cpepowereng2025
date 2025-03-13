# hash:sha256:a0d1d56a1940e01e96dc4e7824fb19696231f886439a54b82e00c4b63c56e3c0
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install -U --no-cache-dir \
    matplotlib==3.9.4 \
    opendssdirect-py==0.9.4 \
    ray[rllib]==2.37.0 \
    torch==2.6.0
