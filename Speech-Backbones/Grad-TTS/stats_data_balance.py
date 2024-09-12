import matplotlib.pyplot as plt

# 初始化字典
spk_dict = {}
acc_dict = {}

# 读取文件并统计
trainfile = 'resources/filelists/zh_all/train.dedup.txt'
with open(trainfile, 'r', encoding='utf8') as input_file:
    for line in input_file:
        uid, text, spk, acc = line.strip().split('|')
        spk_dict[spk] = spk_dict.get(spk, 0) + 1
        acc_dict[acc] = acc_dict.get(acc, 0) + 1

# 打印统计结果
print("acc_dict:", acc_dict)
print("spk_dict:", spk_dict)

# 画柱状图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 绘制 acc_dict 的柱状图
ax1.bar(acc_dict.keys(), acc_dict.values())
ax1.set_title('Distribution of Accents')
ax1.set_xlabel('Accent')
ax1.set_ylabel('Count')
ax1.set_xticklabels(acc_dict.keys(), rotation=45)

# 绘制 spk_dict 的柱状图
ax2.bar(spk_dict.keys(), spk_dict.values())
ax2.set_title('Distribution of Speakers')
ax2.set_xlabel('Speaker')
ax2.set_ylabel('Count')
ax2.set_xticklabels(spk_dict.keys(), rotation=45)

# 调整布局
plt.tight_layout()
plt.savefig('data_distribution_dedup.png')
