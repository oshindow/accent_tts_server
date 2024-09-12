from flask import Flask, request, render_template, send_file, url_for
import subprocess
import os
import time
import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write
import sys
sys.path.append('/home/xintong/accent_tts_server/Speech-Backbones/Grad-TTS/')
from utils import write_hdf5
import torch
import os
import params

from model import GradTTSConformer, GradTTSGST, GradTTSConformerGST
from text import text_to_sequence, text_to_sequence_zh, cmudict, zhdict
from text.symbols import symbols
from utils import intersperse
import os
import sys
sys.path.append('/home/xintong/accent_tts_server/Speech-Backbones/Grad-TTS/hifi-gan/')
# from env import AttrDict
# from models import Generator as HiFiGAN
import librosa
from meldataset import mel_spectrogram, mel_spectrogram_align

from preprocess import clean_and_split, text_to_pinyin, load_lexicon
import yaml
import tqdm
from parallel_wavegan.datasets import (
    AudioDataset,
    AudioSCPDataset,
    MelDataset,
    MelF0ExcitationDataset,
    MelSCPDataset,
)
from parallel_wavegan.utils import load_model, read_hdf5
import soundfile as sf

app = Flask(__name__)

# Load the model when the Flask application starts
def load_models():
    # Load your model here
    # print("Loading model...")
    print('Initializing Grad-TTS...')
    params.n_spks = 222
    n_accents = 1
    params.n_enc_channels = 256
    zh_dict = zhdict.ZHDict('/home/xintong/accent_tts_server/Speech-Backbones/Grad-TTS/resources/zh_dictionary.json')
    # print(zh_dict.__len__())
    generator = GradTTSConformerGST(zh_dict.__len__() + 1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale, n_accents, grl=False, gst=True, cln=True)
    print(generator)
    checkpoint_path = "/data2/xintong/tts_server/Grad-TTS/new_exp_sg_acc_blank_conformer_gst_E8/grad_300.pt"
    checkpoint = torch.load(checkpoint_path, map_location=lambda loc, storage: loc)
    generator.load_state_dict(checkpoint['model'])
    generator.to('cuda:0')
    output_dir = "/data2/xintong/tts_server/ParallelWaveGAN/dump/magichub_sg_16k_gen/eval/gen_grad_300_E8_test/raw"
    print("GradTTS loaded.")
    
    print('Initializing Vocoder...')
    # load config
    vocoder_checkpoint_path = '/data2/xintong/tts_server/ParallelWaveGAN/exp/magichub_sg_16k_csmsc_aishell3_base_finetuning/checkpoint-50000steps.pkl'
    # if args.config is None:
    dirname = os.path.dirname(vocoder_checkpoint_path)
    vocoder_config = os.path.join(dirname, "config.yml")
    with open(vocoder_config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    # config["outdir"] = output_dir

    # check arguments
    # if (args.scp is not None and args.dumpdir is not None) or (
    #     args.scp is None and args.dumpdir is None
    # ):
    #     raise ValueError("Please specify either --dumpdir or --feats-scp.")

    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    vocoder = load_model(vocoder_checkpoint_path, config)
    print(f"Loaded model parameters from {vocoder_checkpoint_path}.")
    # if args.normalize_before:
    #     assert hasattr(model, "mean"), "Feature stats are not registered."
    #     assert hasattr(model, "scale"), "Feature stats are not registered."
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)
    vocoder.to(device)
    
    return generator, zh_dict, output_dir, vocoder, config

    
@app.before_first_request
def before_first_request():
    global generator
    global zh_dict
    global output_dir
    global vocoder
    global config

    generator, zh_dict, output_dir, vocoder, config = load_models()
    
    return generator, zh_dict, output_dir, vocoder, config

def infer_text_to_audio(text):
    global generator
    global zh_dict
    global output_dir
    texts = [text]
    print(texts)
    utt_ids = ["prof_test_sg_acc3"]
    spk_ids = [219]
    acc_ids = [3]
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            # print(texts)
            print(f'Synthesizing {i} \n uid', utt_ids[i], '\n spkid', spk_ids[i], '\n accid', acc_ids[i], '\n',text)
            x = text_to_sequence_zh(text, dictionary=zh_dict)
            x = torch.LongTensor(intersperse(x, len(zh_dict))).cuda()[None]
            # print(x)
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
            spk = torch.LongTensor([spk_ids[i]]).cuda()
            acc = torch.LongTensor([acc_ids[i]]).cuda()
            filepath = '/data2/xintong/magichub_singapore/wav_16k/G0001/A0001_S006_0_G0001_segment_0173.wav'
            
            print('using ' +  filepath.split('/')[-1] + ' as the reference audio')
            
            audio, sr = librosa.load(filepath, sr=16000)
            # print(audio, audio_l, sr_l)
            audio = torch.from_numpy(audio).unsqueeze(0)
            # assert sr == sampling_rate

            # 1. cancel padding inside mel_spectrogram, 2. set center=True, and 3. np.log10
            # then gradtts same as parallel wavegan
            mel = mel_spectrogram_align(audio, 1024, 80, 16000, 256,
                                    1024, 80, 7600, center=True).squeeze()
            
            # print(mel.shape, mel.max(), mel.min())
            y = mel.unsqueeze(0).cuda()
            y_lengths = torch.LongTensor([y.shape[1]]).cuda()
            # print(y.shape, y_lengths)
            t = dt.datetime.now()
            # print(x, spk)
            # print(x, x.shape, x_lengths, spk, acc)
            y_enc, y_dec, attn = generator.forward(x, x_lengths, y, y_lengths, n_timesteps=10, temperature=1.5,
                                                   stoc=False, spk=spk, acc=acc, length_scale=0.91)
            y_dec = y_dec.squeeze(0).transpose(0, 1).cpu().numpy()
            # print(y_dec.max(), y_dec.min())
            # y_dec = np.exp(y_dec)
            # y_dec = np.log10(y_dec)
            print(y_dec.shape, y_dec.max(), y_dec.min())

            
            write_hdf5(
                os.path.join(output_dir, f"{utt_ids[i]}.h5"),
                "feats",
                # mel.transpose(0,1).cpu().numpy().astype(np.float32),
                y_dec.astype(np.float32),
            )
            audio = torch.rand(y_dec.shape[0] * 256)
            # print(torch.max(audio), torch.min(audio))
            write_hdf5(
                os.path.join(output_dir, f"{utt_ids[i]}.h5"),
                "wave",
                audio.cpu().numpy().astype(np.float32),
            )


def infer_mel_to_audio(dumpdir):

    
    mel_query = "*.h5"
    mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA

    dataset = MelDataset(
        dumpdir,
        mel_query=mel_query,
        mel_load_fn=mel_load_fn,
        return_utt_id=True,
    )
    
    
    # logging.info(f"The number of features to be decoded = {len(dataset)}.")

    # start generation
    total_rtf = 0.0
    with torch.no_grad():
        for idx, items in enumerate(dataset):
            # if not use_f0_and_excitation:
            utt_id, c = items
            f0, excitation = None, None
            # print(utt_id, c)
            # else:
            #     utt_id, c, f0, excitation = items
            batch = dict(normalize_before=False)
            if c is not None:
                c = torch.tensor(c, dtype=torch.float).to('cuda:0')
                batch.update(c=c)
            if f0 is not None:
                f0 = torch.tensor(f0, dtype=torch.float).to('cuda:0')
                batch.update(f0=f0)
            if excitation is not None:
                excitation = torch.tensor(excitation, dtype=torch.float).to('cuda:0')
                batch.update(excitation=excitation)
            start = time.time()
            y = vocoder.inference(**batch).view(-1)
            # print(config["sampling_rate"])
            rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
            # pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            # save as PCM 16 bit wav file
            print(config["outdir"])
            sf.write(
                os.path.join(config["outdir"], f"{utt_id}_gen.wav"),
                y.cpu().numpy(),
                config["sampling_rate"],
                "PCM_16",
            )

    # report average RTF
    print(
        f"Finished generation of {idx} utterances (RTF = {total_rtf:.03f})."
    )
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the text from the form
        text = request.form['text']
        
        # Define the path for the output audio file
        dir = "/data2/xintong/tts_server/ParallelWaveGAN/exp/"
        vocoder = "train_nodev_16k_csmsc_parallel_wavegan.v1.16k.finetuning/wav/checkpoint-50000steps"
        output_audio_path = os.path.join(dir, vocoder, 'magichub_sg_16k_gen/eval/gen_grad_300_E8_test/prof_test_sg_acc3_gen.wav')
        config["outdir"] = os.path.dirname(output_audio_path)
        # 0. preprocess
        processed_text = clean_and_split(text)
        lexicon_path = 'lexicon_kaldi.txt'
        lexicon = load_lexicon(lexicon_path)
        pinyin_text = text_to_pinyin(processed_text, lexicon)

        # 1. Grad-TTS
        infer_text_to_audio(pinyin_text)
        # 2. Vocoder
        # subprocess.run(['./run.sh'], check=True)
        infer_mel_to_audio(dumpdir="/data2/xintong/tts_server/ParallelWaveGAN/dump/magichub_sg_16k_gen/eval/gen_grad_300_E8_test/raw")

        # Render the template and pass the URL for the audio file
        subprocess.run(['cp', output_audio_path, 'static/output'])
        audio_url = url_for('static', filename='output/prof_test_sg_acc3_gen.wav', t=time.time())
        return render_template('index.html', audio_file=audio_url)

    
    return render_template('index.html')

@app.route('/download')
def download():
    # Define the path for the output audio file
    dir = "/data2/xintong/tts_server/ParallelWaveGAN/exp/"
    vocoder = "train_nodev_16k_csmsc_parallel_wavegan.v1.16k.finetuning/wav/checkpoint-50000steps"
    output_audio_path = os.path.join(dir, vocoder, 'magichub_sg_16k_gen/eval/gen_grad_300_E8_test/prof_test_sg_acc3_gen.wav')
    
    # Allow the user to download the file
    return send_file(output_audio_path, as_attachment=True)

if __name__ == "__main__":
    before_first_request()
    app.run(host='0.0.0.0', port=8080, debug=True)
