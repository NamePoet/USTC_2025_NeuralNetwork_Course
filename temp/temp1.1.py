import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 进行相关性分析，选择相关性较高的特征，减少过拟合
def select_features_by_correlation(data, target_column='MEDV', threshold=0.4):
    """ 选择与目标变量相关性较高的特征 """
    df = pd.DataFrame(data, columns=[ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ])
    corr_matrix = df.corr()  # 计算相关性矩阵
    target_corr = corr_matrix[target_column].abs().sort_values(ascending=False)  # 目标变量相关性排序

    # 选取高于阈值的特征
    selected_features = target_corr[target_corr > threshold].index.tolist()
    if target_column in selected_features:
        selected_features.remove(target_column)  # 移除目标列本身

    print("Selected Features:", selected_features)

    # 绘制相关性热图
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[selected_features + [target_column]].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    return selected_features


def load_data():
    datafile = 'BostonHousingData.xlsx'
    df = pd.read_excel(datafile, dtype=np.float32)
    data = df.to_numpy()

    if data.shape[0] != 506:
        raise ValueError(f"Dataset should have 506 rows, but found {data.shape[0]}.")

    # 相关性分析
    selected_features = select_features_by_correlation(data)
    selected_indices = [df.columns.get_loc(f) for f in selected_features]  # 获取索引

    # 仅选择相关性较高的特征
    data = data[:, selected_indices + [-1]]  # 选取主要特征 + 房价（MEDV）

    # 使用前450条作为训练集
    train_data = data[:450]

    # 归一化（基于训练集的最大/最小值）
    max_values = train_data.max(axis=0)
    min_values = train_data.min(axis=0)

    for i in range(data.shape[1]):
        if max_values[i] == min_values[i]:
            raise ValueError(f"Feature {i} has identical min and max values, causing division by zero.")
        data[:, i] = (data[:, i] - min_values[i]) / (max_values[i] - min_values[i])

    # 归一化后再分配数据
    train_data = data[:450]     # 训练集：前450条
    test_data = data[456:506]   # 测试集：取最后50条

    return train_data, test_data, max_values, min_values, selected_features


def show_plt(origin, predict):
    plt.plot(origin, color='blue', label='Original Data', alpha=0.6)
    plt.plot(predict, color='red', label='Predicted Data', linewidth=2)
    plt.title('Comparison of Original Data and Predicted Data')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


class Regressor(nn.Module):
    def __init__(self, input_size):
        super(Regressor, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

    def fit_model(self, train_data, num_epochs, batch_size=10):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            mini_batches = [train_data[k:k + batch_size] for k in range(0, len(train_data), batch_size)]
            for mini_batch in mini_batches:
                x = torch.tensor(mini_batch[:, :-1], dtype=torch.float32)
                y = torch.tensor(mini_batch[:, -1:], dtype=torch.float32)
                outputs = self(x)
                loss = criterion(outputs, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


train_data, test_data, max_values, min_values, selected_features = load_data()
# model = Regressor(13)
model = Regressor(len(selected_features))  # 只使用挑选出的特征
num_epochs = 20
batch_size = 10
model.fit_model(train_data, num_epochs, batch_size)

torch.save(model.state_dict(), 'boston_model_6features.pth')
print("Model saved to boston_model.pth")


def predict_pytorch():
    # loaded_model = Regressor(13)
    loaded_model = Regressor(len(selected_features))
    loaded_model.load_state_dict(torch.load('boston_model_6features.pth'))
    loaded_model.eval()

    x_test = torch.tensor(test_data[:, :-1], dtype=torch.float32)
    y_true = test_data[:, -1]

    with torch.no_grad():
        y_pred = loaded_model(x_test).numpy().flatten()

    y_pred = y_pred * (max_values[-1] - min_values[-1]) + min_values[-1]
    y_true = y_true * (max_values[-1] - min_values[-1]) + min_values[-1]

    show_plt(y_true, y_pred)


if __name__ == '__main__':
    predict_pytorch()
