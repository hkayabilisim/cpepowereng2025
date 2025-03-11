# hash:sha256:e6a9fb3c5a2e91be4fd8480fdf41c04bcd15a78d4f8ebb5bf7b2bc8f0d27842c
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install -U --no-cache-dir \
    ray[rllib]==2.43.0
