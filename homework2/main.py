# PyTorch核心模块(CUDA 12.4版本)
import torch
import torch.nn as nn
import torch.optim as optim

# torchvision模块：用于加载数据、数据增强
import torchvision
import torchvision.transforms as transforms

# 科学计算与可视化模块
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 模型评估指标函数
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, classification_report

# 数据拆分工具
from torch.utils.data import random_split, DataLoader


# 数据预处理：转换为Tensor并标准化（均值0.5，方差0.5）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载原始训练数据集
full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 划分训练集和验证集（按8:2划分）
train_size = int(0.8 * len(full_trainset))  # 80% 作为训练集
val_size = len(full_trainset) - train_size  # 20% 作为验证集
trainset, valset = random_split(full_trainset, [train_size, val_size])

# 加载划分后的数据
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

# 提取 CIFAR-10 的标签类别
classes = full_trainset.classes

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            # 第一层：卷积 + 归一化 + 激活 + 池化
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第二层：卷积 + 归一化 + 激活 + 池化
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第三层：卷积 + 归一化 + 激活 + 池化
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        #　全连接层设置
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), # 输出维度根据输入图片大小和卷积层调整
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10) # CIFAR-10 将输出标签分为10类
        )
    # 前向传播函数
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    # CUDA 可用时进行 GPU 加速训练过程
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)

    # 交叉熵损失函数用于分类问题
    criterion = nn.CrossEntropyLoss()

    # Adam 优化器 + 学习率调度器（每10轮衰减）
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    EPOCHS = 15  # 训练轮次为15轮
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []

    val_loss_list = []
    val_acc_list = []

    for epoch in range(EPOCHS):  # 20轮次训练
        model.train()
        running_loss = 0.0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 清空梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_train_loss = running_loss / len(trainloader)
        train_loss_list.append(avg_train_loss)

        # 验证过程
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # 遍历测试集获取预测与真实标签
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(valloader)
        val_accuracy = correct / total
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(val_accuracy)

        # 输出每一轮次的训练损失、验证集损失和验证集准确率
        print(
            f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")


epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(12,5))

# 绘制损失曲线（Loss曲线）
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_list, label='Train Loss')
plt.plot(epochs, val_loss_list, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

# 绘制验证准确率曲线（Accuracy曲线）
plt.subplot(1, 2, 2)
plt.plot(epochs, val_acc_list, label='Val Accuracy', color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

torch.save(model.state_dict(), "cnn_model.pth")

model.eval()
y_true = []
y_pred = []

# 加载测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true += labels.cpu().tolist()
        y_pred += predicted.cpu().tolist()

# 输出总体准确率
acc = accuracy_score(y_true, y_pred)
print(f"\n Test Accuracy: {acc:.4f}")

# 输出每类的 precision
for i, c in enumerate(classes):
    p = precision_score(y_true, y_pred, labels=[i], average='macro')
    print(f"Class {c} Precision: {p:.4f}")

# 输出完整分类报告
print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes))


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()