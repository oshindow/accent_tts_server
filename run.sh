#!/bin/bash

# if [ -z "$1" ]; then
#   echo "Usage: $0 -r <text_string>"
#   exit 1
# fi

# result=$(python preprocess.py "$1")
# pinyin_result=$(echo "$result" | grep 'Pinyin text:' | cut -d':' -f2 | xargs)
# cd /home/xintong/accent_tts_server/Speech-Backbones/Grad-TTS
# bash ./toGPU4.sh "$pinyin_result"
cd /home/xintong/accent_tts_server/ParallelWaveGAN/egs/csmsc/voc1
bash ./tts_pipeline.sh