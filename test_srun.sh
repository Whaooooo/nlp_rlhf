#!/bin/bash

srun --gpus=tesla:2 --cpus-per-gpu=16 --mem=100G --nodelist=frl8a141 \
     --container-image=llm/llm-gpu \
     --container-mounts=/data:/data,/hddlustre:/hddlustre,/lustre:/lustre --container-mount-home \
     --pty bash