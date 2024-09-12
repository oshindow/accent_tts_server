import os

dataset = '/data2/xintong/magichub_singapore/wav_16k'
# waves = {'G0001':[], 'G0002':[], 'G0003':[], 'G0004':[]}
spk2id = {'G0001':218, 'G0002':219, 'G0003':220, 'G0004':221}
accentid = 3
output = open('/home/xintong/Speech-Backbones/Grad-TTS/resources/filelists/zh_all/train_sg.txt', 'w', encoding='utf8')
with open('/home/xintong/Speech-Backbones/Grad-TTS/resources/filelists/magichub_sg/raw.txt', 'r', encoding='utf8') as input:
    for line in input:
        # A0002_S002_0_G0003_segment_0088|sil ee e4 r ang4 r en2 j ia1 j iu3 d eng3 zh ix1 l ei4 d i2 sil|75 7 3 3 7 3 6 10 7 12 14 8 13 10 7 8 8 5 7 23|/data2/xintong/magichub_singapore/mels_2/G0003/A0002_S002_0_G0003_segment_0088.npy|/data2/xintong/magichub_singapore/f0s_1/G0003/A0002_S002_0_G0003_segment_0088.npy|2|228|0
        spk = line.strip().split('|')[0].split('_')[3]
        spkid = spk2id[spk]
        wavepath = os.path.join(dataset, spk, line.strip().split('|')[0] + '.wav')
        phonemes = line.strip().split('|')[1]

        content = wavepath + '|' + phonemes + '|' + str(spkid) + '|' + str(accentid)
        
        output.write(content + '\n')

output.close()

