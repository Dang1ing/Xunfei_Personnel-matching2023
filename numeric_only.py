import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 加载数据集
data = pd.read_csv("data.csv")  # 请替换成您的数据文件路径

# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 数据预处理
scaler = StandardScaler()
numeric_features = ["工作经历数量", "社会经历数量", "项目数量", "技能数量", "荣誉数量"]
train_data[numeric_features] = scaler.fit_transform(
    train_data[numeric_features])
test_data[numeric_features] = scaler.transform(test_data[numeric_features])

# 重新映射岗位ID为连续整数编号
unique_positions = train_data['岗位ID'].unique()
position_mapping = {pos: idx for idx, pos in enumerate(unique_positions)}
train_data['岗位ID'] = train_data['岗位ID'].map(position_mapping)
test_data['岗位ID'] = test_data['岗位ID'].map(position_mapping)

# 构建深度模型


class DeepMatchingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepMatchingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 定义模型参数
input_dim = len(numeric_features)  # 输入特征维度
hidden_dim = 128  # 隐藏层维度
output_dim = len(unique_positions)  # 输出维度，根据新的岗位ID编号数量确定

# 创建模型实例
model = DeepMatchingModel(input_dim, hidden_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 准备训练数据
train_x = torch.tensor(
    train_data[numeric_features].values, dtype=torch.float32)
train_y = torch.tensor(train_data['岗位ID'].values, dtype=torch.long)

# 训练模型
num_epochs = 50
batch_size = 64

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i in range(0, len(train_x), batch_size):
        optimizer.zero_grad()
        batch_x = train_x[i:i+batch_size]
        batch_y = train_y[i:i+batch_size]
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}")

# 在测试集上评估模型
model.eval()
test_x = torch.tensor(test_data[numeric_features].values, dtype=torch.float32)
test_y = torch.tensor(test_data['岗位ID'].values, dtype=torch.long)

with torch.no_grad():
    test_outputs = model(test_x)
    predicted_labels = torch.argmax(test_outputs, dim=1).numpy()

accuracy = accuracy_score(test_y.numpy(), predicted_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
