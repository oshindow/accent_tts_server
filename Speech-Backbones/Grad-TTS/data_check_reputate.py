filepath_dict = {}
with open('/home/xintong/Speech-Backbones/Grad-TTS/resources/filelists/zh_all/train.dedup.txt', 'r', encoding='utf8') as input:
    for line in input:
        filepath, text, spk, acc = line.strip().split('|')
        if filepath not in filepath_dict:
            filepath_dict[filepath] = 1
        else:
            filepath_dict[filepath] += 1

sorted_items = dict(sorted(filepath_dict.items(), key=lambda item: item[1]))

print([(key, value) for key, value in sorted_items.items() if value > 1])