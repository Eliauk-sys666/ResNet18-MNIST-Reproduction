import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import time

# 1. 数据准备
transform = transforms.Compose([
    transforms.Resize((32, 32)), # ResNet 需要 32x32 的输入
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set  = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=1000, shuffle=False)

# 2. 模型定义：修改 ResNet18 以适配单通道，并可选 Dropout
class ResNet18_MNIST(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()
        self.model = resnet18(weights=None) # 从头开始训练
        # 第一层卷积改成 1 通道
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        num_ftrs = self.model.fc.in_features
        if use_dropout:
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.5),         # 提出方法：Dropout
                nn.Linear(num_ftrs, 10)
            )
        else:
            self.model.fc = nn.Linear(num_ftrs, 10) # Baseline：无 Dropout

    def forward(self, x):
        return self.model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("使用设备:", device)

# =======================================================
# 🌟 核心开关：
# 复现 E1 (Baseline): USE_DROPOUT = False, 注释掉 AdamW 和 Cosine
# 复现 E2 (Dropout):  USE_DROPOUT = True,  注释掉 AdamW 和 Cosine
# 复现 E3 (Modern):   USE_DROPOUT = True,  保留 AdamW 和 Cosine (当前状态)
# =======================================================
USE_DROPOUT = True 

if USE_DROPOUT:
    print("正在训练 Proposed 方法 (ResNet-18 + Dropout)...")
    model = ResNet18_MNIST(use_dropout=True).to(device)
    save_name = 'resnet18_modern_proposed.pth'
else:
    print("正在训练 Baseline (标准 ResNet-18)...")
    model = ResNet18_MNIST(use_dropout=False).to(device)
    save_name = 'resnet18_baseline.pth'

criterion = nn.CrossEntropyLoss()

# =======================================================
# 现代化训练策略 (E3 核心)
# 如果你要复现 E1/E2，请把下面两行换成普通的 Adam：
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# 且删掉 scheduler 相关的代码
# =======================================================
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) 
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)     

# 3. 训练与测试 (跑 3 个 Epoch)
def train(epochs=3):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # ❌ 错误位置：scheduler.step() 不能在这里
            
        # ✅ 正确位置：在每个 Epoch 结束后更新学习率
        scheduler.step() 
            
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total   += y.size(0)
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, "
              f"test_acc={correct/total*100:.2f}%")

start = time.time()
train(3)
print(f"总耗时: {time.time()-start:.1f}s")
torch.save(model.state_dict(), save_name)
print(f"权重已保存: {save_name}")