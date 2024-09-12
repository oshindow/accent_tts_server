spk_losses = []
import numpy as np
with open('sg_acc.blank.conformer.gstloss.cln.grl.E14.log', 'r', encoding='utf8') as input:
    for line in input:
        if line.startswith('Epoch: 12,'):
            line = line.strip().split(',')
            spk_loss = float(line[4].split(':')[-1])
            spk_losses.append(spk_loss)
print(max(spk_losses),min(spk_losses),np.mean(spk_losses))
# print()
import matplotlib.pyplot as plt
x = list(range(len(spk_losses)))

# 绘制曲线图
plt.plot(x, spk_losses, marker='o', linestyle='-', color='b')

# 添加标题和标签
plt.title('List Data Plot')
plt.xlabel('Index')
plt.ylabel('Value')    
plt.savefig('spk_losses.png')        
