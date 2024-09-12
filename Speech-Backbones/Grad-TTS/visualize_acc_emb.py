# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write
from utils import write_hdf5
import torch
import os
import params
from model import GradTTS
from text import text_to_sequence, text_to_sequence_zh, cmudict, zhdict
from text.symbols import symbols
from utils import intersperse

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#  2009  python dump_feats_to_GPU4_zh.py -f resources/filelists/synthesis_zh.txt -c logs/new_exp_sg/grad_1053.pt -o logs/new_exp_sg/gen_grad_1053/raw
#  2010  rsync --info=progress2 /home/xintong/Speech-Backbones/Grad-TTS/logs/new_exp_sg/gen_grad_1053/raw xintong@smc-gpu4.d2.comp.nus.edu.sg:/home/xintong/ParallelWaveGAN/egs/csmsc/voc1/dump/magichub_sg_16k_gen/eval/gen_grad_1053/ -r
# python dump_feats_to_GPU4.py -f resources/filelists/synthesis.txt -c logs/new_exp/grad_1518.pt -o logs/new_exp/gen_grad_1518/raw
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=False, default='resources/filelists/synthesis_zh_acc.txt', help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, required=False, default='logs/new_exp_sg_acc/grad_25.pt', help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10, help='number of timesteps of reverse diffusion')
    parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None, help='speaker id for multispeaker model')
    parser.add_argument('-o', '--output_dir', type=str, required=False, default='logs/new_exp_sg/gen_grad_acc_25/raw', help='speaker id for multispeaker model')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # if not isinstance(args.speaker_id, type(None)):
    #     assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
    #     spk = torch.LongTensor([args.speaker_id]).cuda()
    # else:
    #     spk = None
    
    print('Initializing Grad-TTS...')
    params.n_spks = 222
    n_accents = 4
    zh_dict = zhdict.ZHDict('./resources/zh_dictionary.json')
    # print(zh_dict.__len__())
    generator = GradTTS(zh_dict.__len__(), params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale, n_accents)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
    _ = generator.cuda().eval()
    print(generator)
    print(f'Number of parameters: {generator.nparams}')
    
    # print('Initializing HiFi-GAN...')
    # with open(HIFIGAN_CONFIG) as f:
    #     h = AttrDict(json.load(f))
    # vocoder = HiFiGAN(h)
    # vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    # _ = vocoder.cuda().eval()
    # vocoder.remove_weight_norm()
    texts = []
    utt_ids = []
    spk_ids = []
    acc_ids = []
    with open(args.file, 'r', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip().split('|')[1])
            utt_ids.append(line.strip().split('|')[0])
            spk_ids.append(int(line.strip().split('|')[2]))
            acc_ids.append(int(line.strip().split('|')[3]))
    print(spk_ids, acc_ids)

    
    # print(utt_ids, texts)
    visualize_embeddings = [0,0,0,0]
    with torch.no_grad():
        for i, text in enumerate(texts):
            
            print(f'Synthesizing {i} text...', utt_ids[i], text, spk_ids[i])
            
            spk = torch.LongTensor([spk_ids[i]]).cuda()
            acc = torch.LongTensor([acc_ids[i]]).cuda()
            acc_embedding = generator.acc_emb(acc).squeeze(0)
            spk_embedding = generator.spk_emb(spk).squeeze(0)


            if acc_ids[i] == 0:
                visualize_embeddings[0] = acc_embedding.cpu().numpy()
            elif acc_ids[i] == 1:
                visualize_embeddings[1] = acc_embedding.cpu().numpy()
            elif acc_ids[i] == 2:
                visualize_embeddings[2] = acc_embedding.cpu().numpy()
            elif acc_ids[i] == 3:
                visualize_embeddings[3] = acc_embedding.cpu().numpy()
    
    visualize_embeddings = np.array(visualize_embeddings)
    tsne = TSNE(n_components=2, random_state=42)
    accent_embeddings_2d = tsne.fit_transform(visualize_embeddings)

    # Visualize the results
    plt.figure(figsize=(10, 8))
    plt.scatter(accent_embeddings_2d[:, 0], accent_embeddings_2d[:, 1], c=['r', 'g', 'b', 'y'], s=100)

    # Annotate the points
    labels = ['Accent 1', 'Accent 2', 'Accent 3', 'Accent 4']
    for i, label in enumerate(labels):
        plt.annotate(label, (accent_embeddings_2d[i, 0], accent_embeddings_2d[i, 1]), fontsize=12, ha='right')

    plt.title('t-SNE visualization of accent embeddings')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.savefig('accent embeddings')
