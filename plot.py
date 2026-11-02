import matplotlib.pyplot as plt

epochs = [1, 2, 3]

# 填入你跑出的三组真实数据
e1_acc = [97.01, 99.01, 98.47]
e2_acc = [97.97, 98.68, 98.55]
e3_acc = [97.30, 98.91, 99.36]  # <-- 修改这一行

# 下面标出最高点的地方也改一下
plt.text(3, 99.36, ' 99.36%', color='red', fontweight='bold')
plt.figure(figsize=(10, 6))

# 画出三条曲线
plt.plot(epochs, e1_acc, 'b-o', label='Baseline (Adam)')
plt.plot(epochs, e2_acc, 'g-s', label='+ Dropout')
plt.plot(epochs, e3_acc, 'r-^', label='Modern (AdamW+Cosine)', linewidth=2.5)

plt.title('ResNet-18 Test Accuracy on MNIST', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)

# 强制 X 轴只显示整数刻度
plt.xticks(epochs)

# 标出最高点
plt.text(3, 99.36, ' 99.36%', color='red', fontweight='bold')
plt.text(2, 99.01, ' 99.01%', color='blue')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.savefig('resnet_comparison.png', dpi=300)
print("图表已保存为 resnet_comparison.png")