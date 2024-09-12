import os
import json
dataset = '/data2/xintong/aishell3/train/wav_16k'
# waves = {'G0001':[], 'G0002':[], 'G0003':[], 'G0004':[]}
# spk2id = {'G0001':0, 'G0002':1, 'G0003':2, 'G0004':3}
spk2accent = json.load(open('resources/spk2accent.json', 'r', encoding='utf8'))
accent2id = json.load(open('resources/accent2id.json', 'r', encoding='utf8'))
accentid_dict = {}
output = open('/home/xintong/Speech-Backbones/Grad-TTS/resources/filelists/zh_all/train.txt', 'w', encoding='utf8')
with open('/home/xintong/Speech-Backbones/Grad-TTS/resources/filelists/zh_all/raw.txt', 'r', encoding='utf8') as input:
    for line in input:
        utt, phones, frame, mel, f0, accentid, spkid, language = line.strip().split('|')
        spkid = int(spkid)
        # accent = int(accentid)
        spk = utt[:7]
        accentid = accent2id[spk2accent[spk]]
        if accentid not in accentid_dict:
            accentid_dict[accentid] = 1 
        else:
            accentid_dict[accentid] += 1 
        # accent = line.strip().split('|')[0].split('_')[3]
        wavepath = os.path.join(dataset, spk, utt + '.wav')
        phonemes = phones

        content = wavepath + '|' + phonemes + '|' + str(spkid) + '|' + str(accentid)
        
        output.write(content + '\n')
print(accentid_dict)
output.close()

