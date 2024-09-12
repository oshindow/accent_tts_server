import json
output = open('resources/libritts_spk2id.json', 'w', encoding='utf8')
spk2id = {}
with open('resources/filelists/libri-tts/train.txt', 'r', encoding='utf8') as input:
    for line in input:
        filepath, text, spk = line.strip().split('|')
        spkname = filepath.split('/')[6]
        # print(spkname, spk)
        if spkname not in spk2id:
            spk2id[spkname] = spk

json.dump(spk2id, output, indent=4)