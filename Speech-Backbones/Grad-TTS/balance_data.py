import random
from collections import defaultdict

# 读取文件并解析数据
trainfile = 'resources/filelists/zh_all/train.dedup.txt'
data = []

with open(trainfile, 'r', encoding='utf8') as input_file:
    for line in input_file:
        data.append(line.strip().split('|'))

# 删除 accid 为 2 的数据，并将 accid 为 3 的数据改为 2
processed_data = []
for entry in data:
    uid, text, spk, acc = entry
    acc = int(acc)
    spk = int(spk)
    if acc == 2:
        continue
    if acc == 3:
        acc = 2
    processed_data.append([uid, text, spk, acc])

# 将数据按 accid 和 spk id 分组
acc_groups = defaultdict(list)
spk_groups = defaultdict(list)
for entry in processed_data:
    uid, text, spk, acc = entry
    acc_groups[acc].append(entry)
    spk_groups[spk].append(entry)

# 找到 accid 为 0, 1, 2 中最小的数量
min_acc_count = min(len(acc_groups[0]), len(acc_groups[1]), len(acc_groups[2]))

print(min_acc_count)
# 平衡 accid 为 0, 1, 2 的数据量
balanced_data = []
for acc in [0, 1, 2]:
    balanced_data.extend(random.sample(acc_groups[acc], min_acc_count))

# 找到 spk id 小于等于 217 和大于 217 的数量
max_spk_count = 0
for spk, entries in spk_groups.items():
    if spk <= 217:
        max_spk_count = max(len(entries), max_spk_count)

print(max_spk_count)
# 平衡 spk id 大于 217 和其他数据量
# balanced_data.extend(random.sample(spk_groups[218], max_spk_count))
# balanced_data.extend(random.sample(spk_groups[219], max_spk_count))
# balanced_data.extend(random.sample(spk_groups[220], max_spk_count))
# balanced_data.extend(random.sample(spk_groups[221], max_spk_count))

# 保存处理后的数据
output_file = 'resources/filelists/zh_all/train_processed.dedup.txt'
with open(output_file, 'w', encoding='utf8') as output_file:
    for entry in balanced_data:
        output_file.write('|'.join(map(str, entry)) + '\n')

print("数据处理完成并保存到", output_file)
