# Step1：引入所需的库和定义参数
# 引入核心库
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# 设置模型和训练的关键参数
MODEL_NAME = "bert-base-uncased"  # 使用 BERT 英文基础模型
MAX_LENGTH = 256                  # 最大文本长度（截断处理）
BATCH_SIZE = 32                   # 每个 batch 包含的样本数量（适配本机 8GB GPU显存）
NUM_EPOCHS = 3                    # 训练轮数
LEARNING_RATE = 2e-5              # 学习率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # 使用 GPU 加速训练速度
print(f"[INFO] 当前使用设备: {DEVICE}")

# 创建目录用于保存模型和图片
os.makedirs("output", exist_ok=True)


# Step2：加载 IMDB 数据集与预训练分词器
# 下载并加载 IMDB 数据集，包含 train/test 两部分
print("\n[INFO] 加载 IMDB 数据集...")
dataset = load_dataset("imdb")
# 加载预训练分词器
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
print("[INFO] 对文本进行分词和编码...")

# 定义分词函数，对输入文本进行截断和填充
def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )
# 批量处理整个训练集和测试集（提高效率）
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# 设置数据格式为 PyTorch 可接受的格式
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# Step3：构建 DataLoader 进行批量训练，加载预训练 BERT 模型并设置优化器
# 使用 PyTorch 的 DataLoader 封装训练和测试数据
print("[INFO] 构建 DataLoader...")
train_loader = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(tokenized_datasets["test"], batch_size=BATCH_SIZE)

# 加载预训练 BERT 模型
print("[INFO] 加载预训练 BERT 模型...")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# 设置学习率调度器
num_training_steps = NUM_EPOCHS * len(train_loader)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


# Step4：训练模型（支持混合精度AMP）
print("[INFO] 开始训练模型...")
model.train()
scaler = torch.amp.GradScaler(device="cuda")
train_losses = []

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch in loop:
        batch["labels"] = batch.pop("label")
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # 更新学习率

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}\n")

# 保存训练后的模型
model.save_pretrained("output/bert_imdb_model")
tokenizer.save_pretrained("output/bert_imdb_model")


# Step5：可视化训练损失曲线，并在测试集上进行模型评估，显示混淆矩阵
plt.figure(figsize=(8, 4))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, marker='o', color='b')
plt.title("Training loss changes with Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average training loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/training_loss_curve.png")
plt.show()

# 模型评估
print("[INFO] 开始模型测试...")
model.eval()
preds, labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        batch["labels"] = batch.pop("label")
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        preds.extend(predictions.cpu().numpy())
        labels.extend(batch["labels"].cpu().numpy())

# 计算准确率
acc = accuracy_score(labels, preds)
print(f"\n[RESULT] 测试集准确率: {acc * 100:.2f}%")

# 显示混淆矩阵
cm = confusion_matrix(labels, preds)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "positive"])
cmd.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("output/confusion_matrix.png")
plt.show()
